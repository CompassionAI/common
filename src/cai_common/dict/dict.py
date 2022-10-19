import os
from enum import Enum

import pyewts
import dask.bag as db
from cai_common.utils import open_globs


class TibetanEncoding(Enum):
    """Choices of encoding for Tibetan: EWTS (Extended Wylie) or Unicode."""
    UNICODE = 1
    EWTS = 2


class TibetanDict:
    """Dictionary object to translate individual words from Tibetan into English. Supports Extended Wylie and Unicode
    Tibetan encodings.

    Attributes:
        dict_glob (iterable(str) or str): If a string then this is a glob for the location of dictionaries after the
            prefix that locates the repo. If an iterable of strings then each string is treated as a glob and the
            results are unioned to get all the dictionary files.  Defaults to the string
            raw_datasets/tibetan-english-dictionaries/*
        glob_exclusions (iterable(str)): Filenames to exclude from the data glob when loading the dictionaries. Use for
            known bad files. NB: this is not a glob.
        separator (char): The separator character in the dictionary files. To the left of the character is Tibetan in
            Extended Wylie encoding and to the right is the target language. Defaults to '|'.
    """

    _dictionary = {}
    _default_encoding = TibetanEncoding.UNICODE
    dict_glob = "raw_datasets/tibetan-english-dictionaries/*"
    glob_exclusions = set()
    separator = '|'

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None,
                 default_encoding=TibetanEncoding.UNICODE):
        """Constructor for a Tibetan dictionary.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains the
                dictionary files. It will be concatenated with dict_glob to get the full path. For example:
                ../../tibert_data
            glob_override (optional): Overrides the postfix file path glob to specify the location of the corpus inside
                the data registry. For example: raw_datasets/tibetan-english-dictionaries/*
        """

        super().__init__()
        if glob_prefix is None:
            glob_prefix = os.environ['CAI_DATA_BASE_PATH']
        if glob_override is not None:
            self.dict_glob = glob_override
        self.dict_glob = os.path.join(glob_prefix, self.dict_glob)
        self._dict_files = open_globs(self.dict_glob, self.glob_exclusions)
        self._ewts_converter = pyewts.pyewts()
        self._default_encoding = default_encoding
        self._load_dicts(self._dict_files)

    def _process_dict_entry(self, dict_entry):
        # Normalize encoding and translate a single Tibetan word
        tibetan_word, translation = dict_entry.split(self.separator)
        if self.default_encoding == TibetanEncoding.UNICODE:
            tibetan_word = self.EWTS_to_Unicode(tibetan_word)
        elif self.default_encoding == TibetanEncoding.EWTS:
            pass
        else:
            raise ValueError("Bad default_encoding: " + self.default_encoding)
        return (tibetan_word, translation)

    def _load_dicts(self, dict_names):
        # Load all dictionary files in dict_names. The dict_names arg should contain full paths.
        self._dictionary = {}
        _dict_bag = db \
            .read_text(dict_names) \
            .filter(lambda dict_entry: self.separator in dict_entry) \
            .map(self._process_dict_entry) \
            .map(lambda args: (args[0], [t.strip() for t in args[1].split(';')])) \
            .foldby(lambda x: x[0], lambda total, x: total + x[1], [], lambda total1, total2: total1 + total2, [])
        self._dictionary = dict(_dict_bag.compute())

    def __getitem__(self, tibetan):
        """Translate a Unicode Tibetan word into the target language.

        Args:
            tibetan (str or list): Either a single Tibetan word or a list of Tibetan words, encoded according to
                default_encoding.

        Returns:
            List of all dictionary translations for the Tibetan word, or a list of lists if multiple words were
            provided, or None if the word is not found."""
        if isinstance(tibetan, list):
            return [self[tw] for tw in tibetan]
        if tibetan[-1] == '་':
            tibetan = tibetan[:-1]
        return self._dictionary.get(tibetan, None)

    def __len__(self):
        """Get the number of words in the dictionary.

        Returns:
            The number of words in the dictionary."""

        return len(self._dictionary)

    def EWTS_to_Unicode(self, wylie_tibetan):
        """Convert an Extended Wylie encoded Tibetan word into Unicode.

        Args:
            wylie_tibetan: A Tibetan string encoded in Extended Wylie.

        Returns:
            The Unicode encoding of the same Tibetan string."""
        return self._ewts_converter.toUnicode(wylie_tibetan)

    def Unicode_to_EWTS(self, unicode_tibetan):
        """Convert a unicode encoded Tibetan word into Extended Wylie.

        Args:
            wylie_tibetan: A Tibetan string encoded in unicode.

        Returns:
            The Extended Wylie encoding of the same Tibetan string."""
        return self._ewts_converter.toWylie(unicode_tibetan)

    def all_tibetan_words(self):
        """Return all the Tibetan words this dictionary can translate.

        Returns:
            An iterator through all the Tibetan words this dictionary can translate.
        """

        return self._dictionary.keys()

    def keys(self):
        """Synonym for all_tibetan_words."""

        return self.all_tibetan_words()

    def items(self):
        """Return all the Tibetan words this dictionary can translate with their English definitions.

        Returns:
            An iterator through all the Tibetan words this dictionary can translate with their English definitions.
        """

        return self._dictionary.items()

    @property
    def dict_files(self):
        """Read-only property for the actual files used for the dictionaries."""

        return self._dict_files

    @property
    def default_encoding(self):
        """TibetanEncoding: Which encoding for Tibetan to use when translating. Can be UNICODE or EWTS. Changing this
            causes the dictionary to recalculate. Defaults to UNICODE.
        """

        return self._default_encoding

    @default_encoding.setter
    def default_encoding(self, encoding):
        self._default_encoding = encoding
        self._load_dicts(self._dict_files)
