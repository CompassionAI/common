import re
import os
import glob
from enum import Enum
import pandas as pd
import dask.bag as db


_english_punctuation = ".?!"


def _load_from_filename(fn):
    # Load the UTF8 encoded file "fn" and return it as a string.
    with open(fn, encoding='utf-8', mode='r') as f:
        return os.path.basename(fn), f.read()


def _split_sentences(folio):
    # Split a folio into sentences along regular punctuation. Only works for English.
    return [sent.strip()
            for sent in re
                .sub(f"[{_english_punctuation}]", lambda match: match.group() + '\u2406', folio)
                .split("\u2406")]


class CorpusSplitType(Enum):
    """Used to specify how to split the corpus files into individual Dask bag elements."""
    BY_FOLIO = 1
    BY_SECTION = 2
    BY_SENTENCE = 3


class CorpusLoader(object):
    """This is the base class for corpus loader objects. It should not be created by itself.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. For example: OpenPecha/P000001/*.txt.
        exclusions (set): Filenames to exclude from the data glob when loading the dataset. Use for known bad files in
            a particular corpus. NB: this is not a glob.
    """

    _df_column_names, _df_meta = None, None
    data_glob = ''
    glob_exclusions = set()

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None):
        """Constructor for a corpus loader.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. Defaults to the value of the
                CAI_DATA_BASE_PATH environment variable
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../my_data/OpenPecha/P000001/*.txt
        """

        super().__init__()
        if glob_prefix is None:
            glob_prefix = os.environ['CAI_DATA_BASE_PATH']
        if glob_override is not None:
            self.data_glob = glob_override
        else:
            self.data_glob = os.path.join(glob_prefix, self.data_glob)

    def _process_bag(self, bag, locators):
        # Apply whatever map functions or other processing to the Dask bag after loading the files and before splitting
        #   the corpus. To be implemented by the child corpora loaders.
        raise NotImplementedError()

    def _make_bag(self, locators):
        # Make a bag, with or without locators, for final loaded outputs.
        bag = db \
            .from_sequence(filter(
                lambda fn: os.path.basename(fn) not in self.glob_exclusions,
                glob.glob(self.data_glob))) \
            .map(_load_from_filename)
        return self._process_bag(bag, locators=locators)

    def to_bag(self, split_type=CorpusSplitType.BY_FOLIO):
        """Return a bag of segmented text, split in the specified way.

        Args:
            split_type (CorpusSplitType): How to split the text into bag elements. Can be BY_FOLIO, BY_SECTION and
                BY_SENTENCE. Defaults to BY_FOLIO. Tibetan can't be split BY_SENTENCE, while English can't be split
                BY_SECTION.

        Returns:
            dask.bag: The Dask bag that contains the processed corpus.
        """

        bag = self._make_bag(locators=False)
        if split_type == CorpusSplitType.BY_SECTION:
            return bag \
                .str.split(' ') \
                .flatten() \
                .str.replace('\n', '')
        elif split_type == CorpusSplitType.BY_FOLIO:
            return bag
        elif split_type == CorpusSplitType.BY_SENTENCE:
            return bag \
                .map(_split_sentences) \
                .flatten()
        else:
            raise ValueError("Wrong split_type: " + str(split_type))

    @property
    def dataframe(self):
        """dask.dataframe: The Dask dataframe that contains the processed corpus and entry metadata."""
        if self._df_meta is not None:
            return self \
                ._make_bag(locators=True) \
                .to_dataframe(meta=pd.DataFrame({
                    col_name: sample_data
                    for col_name, sample_data in zip(self._df_column_names, self._df_meta)}))
        else:
            return self \
                ._make_bag(locators=True) \
                .to_dataframe(columns=self._df_column_names)
