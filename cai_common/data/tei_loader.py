import re
from bs4.element import Comment, NavigableString, Tag, XMLProcessingInstruction
from bs4 import BeautifulSoup, SoupStrainer
from .corpus_loader import CorpusLoader


def _treat_tags(soup, tag_treatments):
    # Apply the tag_treatments dictionary to the XML soup. Return the soup so that it can be fed to other functions.
    for tag_name, tag_treatment in tag_treatments.items():
        for elem in soup.find_all(tag_name):
            if tag_treatment == 'decompose':
                elem.decompose()
            elif tag_treatment == 'unwrap':
                elem.unwrap()
    return soup


def _treat_divs_and_refs(soup, toh_key, num_toh_keys, refs_to_strip):
    # Decompose refs that are not pointing to the passed in Tohoku number and unwrap divs in the XML soup. Return the
    #   soup so that it can be fed to other functions.
    for ref in soup.find_all("ref"):
        if ref.attrs.get('type', None) in refs_to_strip:
            ref.decompose()
        else:
            if ('key' not in ref.attrs and num_toh_keys > 1) and not set(ref.attrs.keys()) == {'target'}:
                raise Exception("No 'key' attribute in ref tag {}, but more than one Tohoku number for the text {}"
                                .format(ref, toh_key))
            if not ('key' not in ref.attrs or ref.attrs.get('key', None) == toh_key) or 'cRef' not in ref.attrs:
                ref.decompose()
    for div in soup.find_all("div"):
        div.unwrap()
    return soup


def _segment_folios(soup, toh_key):
    # Iterate through the processed XML soup in parsing order and segment it into folios using the remaining ref tags.
    #   Return the segmentation.
    space_re = re.compile(r"\s+")
    segmentation, cur_ref, cur_text = [], None, ''
    for elem in soup.next_elements:
        if type(elem) is NavigableString:
            cur_text += space_re.sub(' ', elem.string)
        elif type(elem) is Tag:
            if not elem.name == 'ref' or not elem.attrs.get('type', 'folio') == 'folio':
                raise ValueError("Bad tag: name {}, type {}, tohoku number {}".format(
                    elem.name, elem.attrs.get('type', 'none'), toh_key))
            segmentation.append((cur_ref, cur_text))
            cur_text, cur_ref = '', elem['cRef']
        elif type(elem) is Comment:
            pass
        elif type(elem) is XMLProcessingInstruction:
            pass
        else:
            raise ValueError("Bad tag {}".format(elem))
    segmentation.append((cur_ref, cur_text))
    return segmentation


def _process_fstr(args):
    # Parse the XML given in args and return a segmentation into folios. Args have to be a tuple with the following
    #   contents:
    #
    #   filename, Tohoku number, number of Tohoku numbers in filename, contents of the filename as a string,
    #   a dictionary of tag treatments, a list of ref tag types to strip
    text_name, toh_key, num_toh_keys, fstr, tag_treatments, refs_to_strip = args
    soup = BeautifulSoup(
        fstr,
        'xml',
        from_encoding='UTF8',
        parse_only=SoupStrainer("div", type="translation"))
    return [
        (text_name, toh_key, ref, text)
        for ref, text in _segment_folios(
            _treat_divs_and_refs(
                _treat_tags(soup, tag_treatments), toh_key, num_toh_keys, refs_to_strip), toh_key)]


def _fn_to_tohs(args):
    # Extract the list of Tohoku numbers from the filename. For example:
    #
    #   "072-012_toh312,628,1093-the_mahasutra_entering_the_city_of_vaisali.xml" => ["toh312", "toh628", "toh1093"]
    #
    # Args has to be a tuple of (filename, contents of the filename as a string). The function returns a list of
    #   tuples, one tuple per Tohoku number in the filename, of the form:
    #
    #   [(filename, Tohoku number, number of Tohoku numbers in filename, contents of the filename as a string)]
    fn, fstr = args
    toh_re_res = re.compile(r"_toh[\d,]+[-_]").search(fn)
    if toh_re_res:
        toh_keys = ['toh' + toh_num for toh_num in toh_re_res.group(0)[4:-1].split(',')]
        return [(fn, toh_key, len(toh_keys), fstr) for toh_key in toh_keys]
    else:
        return []


class TeiLoader(CorpusLoader):
    """Corpus loader for the 84,000 TEI translation files. Source repo is https://github.com/84000/all-data.

    Attributes:
        data_glob: A glob for the location of the data after the prefix that locates the repo. The prefix is specified
            in the constructor. Defaults to: 84000/data-tei/translations/{corpus}/translations, you have to specify
            corpus in the constructor, currently can be kangyur or tengyur.
        glob_exclusions (set): Filenames in the corpus to ignore. Defaults to 5 files in the Kangyur, one of which has
            the word OLD in its Tohoku number and the others have no <ref> tags.
        ns: The XML namespace for the TEI tags. Defaults to http://www.tei-c.org/ns/1.0.
        tag_treatments (dictionary): Mapping of XML tags with instructions on how to treat them. There are currently
            two possible tag treatments: unwrap and decompose. The meaning of the operations is as in Beautiful Soup 4.
            NB: The treatments have been carefully set from several days of poring over the dataset. Think twice before
            tinkering with this.
        ref_types_to_strip (set): Reference tags usually (but not always) come with a "type" attribute. This is the
            list of the values of this "type" attribute to remove from the XML. Defaults to bampo, sanskrit and volume,
            all lowercase.
    """

    # Additional columns like volume_number are added after parent class generates _df_column_names
    _df_column_names = ["filename", "tohoku_number", "location", "text"]
    _df_meta = [['a'], ['a'], ['a'], ['a']]
    _df_final_columns = ["filename", "volume_number", "tohoku_number", "location", "text"]

    data_glob = "raw_datasets/84000-translations-tei/translations/{corpus}/translations/*.xml"
    glob_exclusions = {
        '079-008_toh381OLD-the_emergence_from_samputa%20(20190216).xml',
        '040-005_toh54-the_chapter_on_the_complete_approach.xml',
        '043-006_toh69-arousing_determination.xml',
        '043-006_toh69-the_sutra_which_incites_resolve.xml',
        '057-010_toh151-the_sutra_of_the_questions_of_pratibhanamati.xml',
        '061-012_toh192-the_prophecy_of_ksemavati.xml',
        '066-007_toh249-the_sutra_teaching_four_qualities.xml',
        '067-005_toh266-the_sutra_of_the_flower_collection.xml',
        '077-002_toh361-a_summary_explanation_of_the_initiation.xml',
        '094-006_toh729,1001-the_incantation_of_tara.xml'}
    ns = "http://www.tei-c.org/ns/1.0"
    tag_treatments = {
        'term': 'unwrap',
        'lg': 'unwrap',
        'lb': 'decompose',
        'milestone': 'decompose',
        'foreign': 'unwrap',
        'distinct': 'unwrap',
        'emph': 'unwrap',
        'title': 'unwrap',
        'head': 'unwrap',
        'q': 'unwrap',
        'l': 'unwrap',
        'trailer': 'unwrap',
        'note': 'decompose',
        'p': 'unwrap',
        'mantra': 'unwrap',
        'list': 'unwrap',
        'item': 'unwrap',
        'label': 'decompose',
        'canonDef': 'unwrap',
        'media': 'decompose',
        'fix': 'decompose'}
    ref_types_to_strip = {'bampo', 'sanskrit', 'volume'}

    def __init__(self,
                 tei_corpus,
                 glob_prefix=None,
                 glob_override=None):
        """Constructor for a corpus loader for the 84,000 TEI translation files.

        Args:
            glob_prefix (optional): The prefix to the current path for the location of the repo that contains this
                corpus. It will be concatenated with data_glob to get the full path. For example: ../../my_data
            tei_corpus: Which of the 84,000 translated corpora to load. It is used to form the directory that the XML
                files are in: 84000/data-tei/translations/{tei_corpus}/translations. Currently, can be kangyur or
                tengyur.
            glob_override (optional): Overrides the entire file path glob to specify the location of the corpus. For
                example: ../../my_data/84000/data-tei/translations/kangyur/translations/*.xml
        """

        self.data_glob = self.data_glob.format(corpus=tei_corpus)
        super().__init__(
            glob_prefix=glob_prefix,
            glob_override=glob_override)

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Adds the Tohoku numbers, applies folio segmentation,
        #   strips markup, removes spaces, and removes locators if requested.
        bag = bag \
            .map(_fn_to_tohs) \
            .flatten() \
            .map(lambda args: tuple(list(args) + [self.tag_treatments, self.ref_types_to_strip])) \
            .map(_process_fstr) \
            .flatten()
        if not locators:
            bag = bag.map(lambda args: args[-1])
        return bag

    @property
    def dataframe(self):
        res_df = super().dataframe
        res_df["volume_number"] = res_df.filename.map(lambda x: x.split('-')[0]).astype(int)
        res_df = res_df[self._df_final_columns]
        return res_df
