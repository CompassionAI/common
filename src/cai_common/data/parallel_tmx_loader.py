import re
from bs4 import BeautifulSoup

from .corpus_loader import CorpusLoader
from .utils import repl_split_commas_or_kill


_strip_chars = {'!'}
_bad_chars = {
    '_', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y',
    '\u200b'}


def _strip_out_pairs(soup, strip_chars, bad_chars, apply_markup, replace_with_suggested):
    pairs = []
    for sent_pair_tag in soup.find_all("tu"):
        props = sent_pair_tag.find_all("prop")
        folio, position = '', ''
        for prop_tag in props:
            if prop_tag.attrs['name'] == 'folio':
                folio = ' '.join(list(prop_tag.stripped_strings))
            elif prop_tag.attrs['name'] == 'position':
                position = ' '.join(list(prop_tag.stripped_strings))
        cur_pair = [' '.join(list(tag.stripped_strings)) for tag in sent_pair_tag.find_all("tuv")
                    if tag.get('xml:lang') in ['bo', 'en']]
        assert len(cur_pair) == 2
        tibetan, english = cur_pair
        if len(set(tibetan).intersection(bad_chars)) > 0:
            continue
        for c in strip_chars:
            tibetan = tibetan.replace(c, '')
        if apply_markup:
            tibetan = re.sub(
                r"[\[({][^\[\](){}]*?[})\]]",
                lambda x: repl_split_commas_or_kill(x, take_second_item=replace_with_suggested),
                tibetan)
        pairs.append((folio, position, tibetan, english))
    return pairs


def _process(args,            # pylint: disable=dangerous-default-value
             with_locator=True,
             strip_chars=_strip_chars,
             bad_chars=_bad_chars,
             apply_markup=True,
             replace_with_suggested=False):
    fn, fstr = args
    soup = BeautifulSoup(fstr, 'xml')
    cur_pairs = _strip_out_pairs(soup, strip_chars, bad_chars, apply_markup, replace_with_suggested)
    if with_locator:
        toh = re.findall(r"Toh_[\d-]+[a-z]?-", fn)
        if len(toh) == 0:
            raise ValueError(f"Filename has no Tohoku number: {fn}")
        toh = toh[0][4:-1]
        return [(fn, toh, folio, position, tibetan, english) for folio, position, tibetan, english in cur_pairs]
    else:
        return [(tibetan, english) for _, _, tibetan, english in cur_pairs]


class ParallelTMXLoader(CorpusLoader):
    """Loader for the parallel sentences dataset from 84,000. Source repo is
        https://github.com/84000/data-translation-memory.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. Defaults to: raw_datasets/84000-parallel-sentences/*.tmx
        strip_chars (set of characters): Set of characters to delete from the Tibetan strings.
        bad_chars (set of characters): Any Tibetan string that contains a character in this set is removed from the
            dataset and its pair is skipped.
    """

    _df_column_names = ["filename", "tohoku", "folio", "position", "tibetan", "english"]
    _df_meta = [['a'], ['a'], ['a'], ['a'], ['a'], ['a']]
    _apply_markup = True
    _replace_with_suggested = False
    _clean_bad_chars = True

    data_glob = "raw_datasets/84000-parallel-sentences/*.tmx"
    glob_exclusions = {
        "Modular_Passages-TM-Test.tmx"
    }
    strip_chars = _strip_chars
    bad_chars = _bad_chars

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None):
        """Constructor for a corpus loader for the parallel sentences dataset from 84,000. The default settings are to
            apply markup, not to replace with suggestions in the markup, and to clean up bad characters.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. For example: ../../my_data
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../my_data/84000/data-translation-memory/*.txt
        """

        super().__init__(glob_prefix=glob_prefix, glob_override=glob_override)

    def apply_markup(self, apply=True, replace_with_suggested=False):
        """Set whether to apply the markup for suggested replacements within the corpus.

        Args:
            apply: Markup is treated if apply=True. If apply=False then the markup is not touched at all and the
                brackets are left as is. Defaults to True.
            replace_with_suggested: If True then the suggested replacements in the markup are performed. This means
                that elements like (x,y) become y. Defaults to False.

        Returns:
            The corpus loader object so that methods can be chained in the functional style.
        """

        self._apply_markup = apply
        self._replace_with_suggested = replace_with_suggested
        return self

    def clean_bad_chars(self, clean=True):
        """Set whether to remove bad characters from the corpus. Each corpus has to specify its own bad character set.

        Args:
            clean: Remove bad characters if True. Defaults to True.

        Returns:
            The corpus loader object so that methods can be chained in the functional style.
        """

        self._clean_bad_chars = clean
        return self

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Applies markup in round and curly braces and removes
        #   bad characters, if requested.
        strip_chars, bad_chars = (self.strip_chars, self.bad_chars) if self._clean_bad_chars else (set(), set())
        bag = bag \
            .map(lambda x: _process(x,
                                    strip_chars=strip_chars,
                                    bad_chars=bad_chars,
                                    apply_markup=self._apply_markup)) \
            .flatten()
        if not locators:
            bag = bag \
                .map(lambda args: args[1:])
        return bag
