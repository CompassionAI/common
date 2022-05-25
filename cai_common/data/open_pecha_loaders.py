import re
from .corpus_loader import CorpusLoader
from cai_common.dict import tibetan_digits, translate_tibetan_number

from .utils import repl_split_commas_or_kill


def _apply_with_locators(args, func, locators):
    # Apply a function to a bag element, taking the presence or absence of locators into account.
    if locators:
        args = list(args)
        return tuple(args[:-1] + [func(args[-1])])
    else:
        return func(args)


class OpenPechaLoader(CorpusLoader):
    """This is the base class for corpus loaders of material from the OpenPecha repositories. It should not be created
    by itself.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. For example: OpenPecha/P000001/*.txt
    """

    _space_re = re.compile(r"\s+")
    _apply_markup = True
    _replace_with_suggested = False
    _clean_bad_chars = True
    _remove_section_headings = True
    _remove_new_lines = False

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None,
                 test_mode=False):
        """Constructor for a corpus loader from an OpenPecha repository. The default settings are to apply markup, not
        to replace with suggestions in the markup, to clean up bad characters, and to remove section headings.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. For example: ../../tibert_data
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../tibert_data/OpenPecha/P000001/*.txt
            test_mode: When applying markup, retain the original and the markup being applied. Formatted as @(x,y)|res@
                where res is either x or y depending on the setting of replace_with_suggested given in apply_markup.
                Defaults to False, intended for testing the markup application.
        """

        super().__init__(
            glob_prefix=glob_prefix,
            glob_override=glob_override)
        self.test_mode = test_mode

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

    def remove_section_headings(self, remove=True):
        """Set whether to remove various headings from the corpus, such as Tohoku catalog numbers or folio numbers.

        Args:
            clean: Remove headings if True. Defaults to True.

        Returns:
            The corpus loader object so that methods can be chained in the functional style.
        """

        self._remove_section_headings = remove
        return self

    def remove_new_lines(self, remove=True):
        """Set whether to remove newline characters from the corpus. This will connect sections split by line breaks
            but not by folio breaks.

        Args:
            clean: Remove newlines if True. Defaults to False.

        Returns:
            The corpus loader object so that methods can be chained in the functional style.
        """

        self._remove_new_lines = remove
        return self

    def _format_folio_locator(self, match):
        raise NotImplementedError()

    def _segment_folios(self, args):
        # Segment folios using the folio marker regular expression by stepping through successive folio marker matches
        #   and taking the text in between. Also converts the Tibetan numerals for page markers into 84,000-style folio
        #   locators of the form F.page.a(verso)/b(recto).
        fn, volume_str = args
        volume_num = int(fn.split("_")[0])
        res, cur_ref, last_loc = [], None, None
        for match in self._folio_marker_re.finditer(volume_str):
            cur_loc_start, cur_loc_end = match.span()
            if last_loc is not None:
                res.append((fn, volume_num, cur_ref, volume_str[last_loc:cur_loc_start]))
            last_loc = cur_loc_end
            cur_ref = self._format_folio_locator(match)
        res.append((fn, volume_num, cur_ref, volume_str[last_loc:]))
        return res

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Applies folio segmentation, removes spaces, and
        #   removes locators if requested.
        bag = bag \
            .map(self._segment_folios) \
            .flatten() \
            .map(
                lambda args: _apply_with_locators(
                    args,
                    lambda x:
                        self._space_re.sub(' ', x.replace('\n', '') if self._remove_new_lines else x).strip(), True))
        if not locators:
            bag = bag \
                .map(lambda args: args[-1]) \
                .filter(lambda x: not x == '')
        return bag


class KangyurLoader(OpenPechaLoader):
    """Esukhia Derge Kangyur corpus loader. Source repo is https://github.com/OpenPecha/P000001.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. Defaults to: OpenPecha/P000001/*.txt
    """

    _df_column_names = ["filename", "volume_number", "location", "text"]
    _df_meta = [['a'], ['a'], ['a'], ['a']]
    _folio_marker_re = re.compile(r"\(([{}]+)([ནབ])་\)".format(''.join(tibetan_digits)))
    data_glob = "OpenPecha/P000001/*.txt"

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None,
                 test_mode=False):
        """Constructor for the Kangyur loader from the OpenPecha repository. The default settings are to apply markup,
        not to replace with suggestions in the markup, to clean up bad characters, and to remove section headings.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. For example: ../../tibert_data
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../tibert_data/OpenPecha/P000001/*.txt
            test_mode: When applying markup, retain the original and the markup being applied. Formatted as @(x,y)|res@
                where res is either x or y depending on the setting of replace_with_suggested given in apply_markup.
                Defaults to False, intended for testing the markup application.
        """

        super().__init__(
            glob_prefix=glob_prefix,
            glob_override=glob_override,
            test_mode=test_mode)

    def _format_folio_locator(self, match):
        # Format the result of the folio marker regular expression (self._folio_marker_re) to be in the standard 84,000
        #   form F.page.a(verso)/b(recto).
        return "F.{}.{}".format(translate_tibetan_number(match.group(1)), 'a' if match.group(2) == 'ན' else 'b')

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Applies folio segmentation, removes spaces, and
        #   removes locators if requested.
        #
        # Basic strategy:
        #    0. First segment into folios if required
        #    1. Find all round and curly braces (no curly braces in the Kangyur but inspired by the Tengyur we still do
        #       it)
        #        a. If there is a comma in the bracketed text, split along comma and take first element (uncorrected)
        #        b. If no comma then just kill entire bracketed text (remove comments and mark-up)
        #    2. Remove square brackets and leave text inside them (presumed interpolation for damage etc)
        #    3. Remove unmatched commas, brackets, stars, the BOM, etc (leaves small number of bad morphemes that are
        #       very hard to clean)
        bag = super()._process_bag(bag, locators)
        if self._apply_markup:
            # Have to repeat twice because some brackets are nested to depth 2
            bag = bag.map(lambda args: _apply_with_locators(
                args,
                lambda x: re.sub(
                    r"[({][^(){}]*?[})]",
                    lambda y: repl_split_commas_or_kill(y, self._replace_with_suggested, self.test_mode),
                    x),
                locators))
            if not self.test_mode:
                bag = bag.map(lambda args: _apply_with_locators(
                    args,
                    lambda x: re.sub(
                        r"[({][^(){}]*?[})]",
                        lambda y: repl_split_commas_or_kill(y, self._replace_with_suggested, self.test_mode),
                        x),
                    locators))
        if self._clean_bad_chars:
            # There are opening/closing brackets with no matching brackets
            bad_chars_to_remove = "*`p}1-9\ufeff�‘’"
            bag = bag.map(lambda args: _apply_with_locators(
                args,
                lambda x: re.sub(r"[\[\]{}]".format(bad_chars_to_remove), '', x),
                locators))
        return bag


class TengyurLoader(OpenPechaLoader):
    """Esukhia Derge Tengyur corpus loader. Source repo is https://github.com/Esukhia/derge-tengyur.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. Defaults to: derge-tengyur/text/*.txt.
    """

    _df_column_names = ["filename", "volume_number", "location", "text"]
    _folio_marker_re = re.compile(r"\[([\d]+)([ab])\]")
    data_glob = "derge-tengyur/text/*.txt"

    def __init__(self,
                 glob_prefix=None,
                 glob_override=None,
                 test_mode=False):
        """Constructor for the Tengyur loader from the Esukhia repository. The default settings are to apply markup,
        not to replace with suggestions in the markup, to clean up bad characters, and to remove section headings.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. For example: ../../tibert_data
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../tibert_data/derge-tengyur/text/*.txt
            test_mode: When applying markup, retain the original and the markup being applied. Formatted as @(x,y)|res@
                where res is either x or y depending on the setting of replace_with_suggested given in apply_markup.
                Defaults to False, intended for testing the markup application.
        """

        super().__init__(
            glob_prefix=None,
            glob_override=glob_override,
            test_mode=test_mode)

    def _format_folio_locator(self, match):
        # Format the result of the folio marker regular expression (self._folio_marker_re) to be in the standard 84,000
        #   form F.page.a(verso)/b(recto).
        return "F.{}.{}".format(match.group(1), match.group(2))

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Applies folio segmentation, removes spaces, and
        #   removes locators if requested.
        #
        # Basic strategy:
        #    1. Find all round and curly braces
        #        a. If there is a comma in the bracketed text, split along comma and take first element (uncorrected)
        #        b. If no comma then just kill entire bracketed text (remove comments and mark-up)
        #    2. Remove text of the form NNNab.NNN where N are numebrs inside square brackets (section/folio/work/etc
        #       numbering)
        #    3. Remove square brackets and leave text inside them (presumed interpolation for damage etc)
        #    4. Remove hashtags (mark-up and comments)
        #    5. Remove unmatched commas, brackets, stars, and the BOM (leaves small number of bad morphemes that are
        #       very hard to clean)
        bag = super()._process_bag(bag, locators)
        if self._apply_markup:
            # Have to repeat twice because some brackets are nested to depth 2
            bag = bag.map(lambda args: _apply_with_locators(
                args,
                lambda x: re.sub(
                    r"[({][^(){}]*?[})]",
                    lambda y: repl_split_commas_or_kill(y, self._replace_with_suggested, self.test_mode),
                    x),
                locators))
            if not self.test_mode:
                bag = bag.map(lambda args: _apply_with_locators(
                    args,
                    lambda x: re.sub(
                        r"[({][^(){}]*?[})]",
                        lambda y: repl_split_commas_or_kill(y, self._replace_with_suggested, self.test_mode),
                        x),
                    locators))
        if self._remove_section_headings:
            bag = bag.map(lambda args: _apply_with_locators(
                args,
                lambda x: re.sub(r"\[[0-9 ]+[ab.0-9]+\]", '', x),
                locators))
        if self._clean_bad_chars:
            # There are opening/closing brackets with no matching brackets
            bad_chars_to_remove = "\ufeff\x07\t+./0-9:;<=>A-Za-z{×"
            bag = bag.map(lambda args: _apply_with_locators(
                args,
                lambda x: re.sub(r"[\[\]#\(\),*{}]".format(bad_chars_to_remove), '', x),
                locators))
        return bag
