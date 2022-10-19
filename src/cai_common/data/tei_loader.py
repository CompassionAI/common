import logging

import re
from bs4.element import Comment, NavigableString, Tag, XMLProcessingInstruction
from bs4 import BeautifulSoup, SoupStrainer
from .corpus_loader import CorpusLoader

logger = logging.getLogger(__name__)

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
        if not "xml:id" in ref.attrs:
            ref.decompose()
        elif ref.attrs.get('type', None) in refs_to_strip:
            ref.decompose()
        else:
            if ('key' not in ref.attrs and num_toh_keys > 1) and not set(ref.attrs.keys()) == {'target'}:
                raise Exception(f"No 'key' attribute in ref tag {ref}, but more than one Tohoku number for the text "
                                f"{toh_key}")
            if not ('key' not in ref.attrs or ref.attrs.get('key', None) == toh_key) or 'cRef' not in ref.attrs:
                ref.decompose()
    for div in soup.find_all("div"):
        div.unwrap()
    return soup


def _folio_str_to_page_num(folio_str):
    # Converts F.a.b to a (relative) page number
    splits = folio_str.split('.')
    return 2 * int(splits[1]) + (1 if splits[2] == 'b' else 0)


def _segment_folios(soup, toh_key, volumes):
    # Iterate through the processed XML soup in parsing order and segment it into folios using the remaining ref tags.
    #   Return the segmentation.
    valid_tag_types = {'folio', 'volume'}
    space_re = re.compile(r"\s+")
    segmentation, cur_volume, vol_start_page, cur_ref, cur_text = [], None, None, None, ''
    for elem in soup.next_elements:
        if isinstance(elem, NavigableString):
            cur_text += space_re.sub(' ', elem.string)
        elif isinstance(elem, Tag):
            elem_type = elem.attrs.get('type', 'folio')
            if not (elem.name == 'ref' and elem_type in valid_tag_types):
                raise ValueError(f"Bad tag: name {elem.name}, type {elem.attrs.get('type', 'none')}, tohoku number "
                                 f"{toh_key}")
            segmentation.append((cur_volume, cur_ref, cur_text))
            if elem_type == 'folio':
                ref_page = _folio_str_to_page_num(elem['cRef'])
                if vol_start_page is None:
                    vol_start_page = ref_page
                if cur_volume is None:
                    if len(volumes) > 1:
                        raise ValueError(f"In {toh_key}, no volume specified in start of doc but more than one volume "
                                          "in metadata.")
                    cur_volume = next(iter(volumes.keys()))
                if not cur_volume in volumes:
                    raise ValueError(f"In {toh_key}, the current volume was {cur_volume} but is not in the volume "
                                     f"metadata ({sorted(list(volumes.keys()))})")
                cur_text, cur_ref = '', ref_page - vol_start_page + volumes[cur_volume]['start_page'] - 1
                if cur_ref > volumes[cur_volume]['end_page']:
                    logger.warning(f"In {toh_key}, the current page was {cur_ref} which is larger than the end page "
                                   f"{volumes[cur_volume]['end_page']} for volume {cur_volume}, "
                                   f"xml:id={elem['xml:id']}")
            elif elem_type == 'volume':
                cur_volume = elem['cRef']
                if not (cur_volume[0] == 'V' and cur_volume[1:].isdigit()):
                    raise ValueError(f"Badly formatted volume number {cur_volume} in tag name {elem.name}, type "
                                     f"{elem.attrs.get('type', 'none')}, tohoku number {toh_key}")
                cur_volume = int(cur_volume[1:])
                vol_start_page = None
        elif isinstance(elem, Comment):
            pass
        elif isinstance(elem, XMLProcessingInstruction):
            pass
        else:
            raise ValueError(f"Bad tag {elem}")
    segmentation.append((cur_volume, cur_ref, cur_text))
    return segmentation


def _process_fstr(args):
    # Parse the XML given in args and return a segmentation into folios. Args have to be a tuple with the following
    #   contents:
    #
    #   filename, Tohoku number, number of Tohoku numbers in filename, contents of the filename as a string,
    #   a dictionary of tag treatments, a list of ref tag types to strip
    text_name, toh_key, num_toh_keys, volumes, fstr, tag_treatments, refs_to_strip = args
    soup = BeautifulSoup(
        fstr,
        'xml',
        parse_only=SoupStrainer("div", type="translation"))
    return [
        (text_name, toh_key, volume, ref, text)
        for volume, ref, text in
            _segment_folios(
                _treat_divs_and_refs(
                    _treat_tags(
                        soup,
                        tag_treatments
                    ),
                    toh_key,
                    num_toh_keys,
                    refs_to_strip
                ),
                toh_key,
                volumes
            )
        ]


def _fn_to_tohs(args):
    # Extract the list of Tohoku numbers from the metadata sourceDesc tag
    #
    # Args has to be a tuple of (filename, contents of the filename as a string). The function returns a list of
    #   tuples, one tuple per Tohoku number in the text, of the form:
    #
    #   [(filename, Tohoku number, count of Tohoku numbers in text, contents of the filename as a string)]
    fn, fstr = args
    soup = BeautifulSoup(
        fstr,
        'xml',
        parse_only=SoupStrainer("sourceDesc")
    )
    toh_metadata = []
    for source in soup.find_all("bibl"):
        volumes = {}
        for volume in source.find_all("volume"):
            volumes[int(volume['number'])] = {
                "start_page": int(volume['start-page']),
                "end_page": int(volume['end-page'])
            }
        toh_metadata.append((source['key'], volumes))
    return [(fn, toh_key, len(toh_metadata), volumes, fstr) for toh_key, volumes in toh_metadata]


def _recto_verso_paginate(args):
    text_name, toh_key, volume, ref, text = args
    p = 0 if ref is None else int(ref)
    ref = f"F.{(p - 1) // 2 + 1}.{'a' if (p % 2) == 0 else 'b'}"
    return (text_name, toh_key, volume, ref, text)


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
    _df_column_names = ["filename", "tohoku_number", "volume_number", "location", "text"]
    _df_meta = [['a'], ['a'], ['a'], ['a'], ['a']]
    _df_final_columns = ["filename", "tohoku_number", "volume_number", "location", "text"]
    _recto_verso_pagination = False

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
        '094-006_toh729,1001-the_incantation_of_tara.xml',
        '094-007_toh730,1002-the_incantation_taras_own_promise.xml',
        '048-001_toh101-upholding_the_roots_of_virtue_backup.xml',
        '048-001_toh101-obtaining_the_roots_of_virtue.xml',
        '096-062_toh813,1098-a_prayer_from_destroyer_of_the_great_chiliocosm_cf_toh_558.xml',
        '057-001_toh142-entering_into_non-conceptuality.xml',
        '058-006_toh157-the_sutra_of_the_questions_of_the_kinnara_king_druma.xml',
        '059-008_toh165-the_sutra_of_the_questions_of_ksemamkara.xml',
        '060-007_toh178-the_sutra_of_the_teaching_on_the_aids_to_enlightenment.xml',
        '060-007_toh178-teaching_on_the_aids_to_enlightenment.xml',
        '061-001_toh181-the_sutra_teaching_the_five_perfections.xml',
        '061-016_toh196-the_sutra_on_the_residence_of_manjusri.xml',
        '061-018_toh198-the_sutra_on_maitreyas_setting_forth.xml',
        '062-003_toh203-the_sutra_of_the_seal_of_the_dharma.xml',
        '062-007_toh207-the_sutra_on_the_elephants_exertions.xml',
        '062-007_toh207-the_the_strength_of_the_elephant.xml',
        '062-011_toh211-the_sutra_teaching_the_beginnings_and_the_divisions_of_dependent_arising.xml',
        '066-002_toh244-the_sutra_of_the_way_of_the_dharma.xml',
        '066-003_toh245-the_sections_of_dharmas.xml',
        '066-003_toh245-the_sutra_of_the_collection_of_dharmas.xml',
        '066-005_toh247-distinguishing_between_dharmas_and_their_reality.xml',
        '066-005_toh247-the_sutra_on_how_to_distinguish_dharma_and_worldly_pursuits.xml',
        '066-010_toh252-the_sutra_on_accomplishing_sets_of_four_things.xml',
        '068-002_toh268-the_s%C5%ABtra_of_king_of_the_inconceivable.xml',
        '068-004_toh270,512,852-the_sutra_of_the_seven_buddhas.xml',
        '068-005_toh271-the_sutra_of_the_eight_buddhas.xml',
        '068-007_toh273,511,853-the_sutra_of_the_twelve_buddhas.xml',
        '068-012_toh278-the_sutra_on_the_eightfold_auspiciousnesses.xml',
        '068-019_toh285-the_dedication_perfecting_all_intentions.xml',
        '072-017_toh317-the_dharma_discourse_on_the_ascertainment_of_the_meaning.xml',
        '072-025_toh325-verses_on_the_naga_king_bheri.xml',
        '075-003_toh345-the_account_of_noble_deeds_concerning_a_sow.xml',
        '088-008_toh514,854-the_incantation_the_essence_of_the_buddhas__a_dharma_discourse.xml',
        '088-009_toh515,855-the_incantation_the_essence_of_the_buddhas.xml',
        '088-016_toh522,848-the_dharani_of_the_torch_of_gnosis.xml',
        '042-002_toh61-the_sutra_of_purnas_questions.xml',
        '042-003_toh62-the_sutra_of_rastrapalas_questions.xml',
        '091-072_toh674,849-the_aparimitayurjnana_sutra.xml',
        '091-072_toh674,849-the_sutra_of_immeasurable_life_and_gnosis.xml',
        '091-074_toh676,850-the_incantation_the_essence_of_immeasurable_life_and_gnosis.xml',
        '091-077_toh679,851-the_incantation_immeasurable_praise_of_good_qualities.xml',
        '043-007_toh70-question_of_subahu.xml',
        '043-007_toh70-the_sutra_of_subahus_questions.xml',
        '094-002_toh725,909-the_incantation_mother_of_avalokitesvara.xml',
        '094-008_toh731-the_sutra_of_tara_who_protects_from_the_eight_fearful_things.xml',
        '094-013_toh736,995-the_incantation_of_parnasabari.xml',
        '100-001_toh846-the_three_part_invitation_tantra.xml',
        '100-002_toh846a-the_threefold_ritual.xml'}
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
    ref_types_to_strip = {'bampo', 'sanskrit'}

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

    def recto_verso_pagination(self, recto_verso=True):
        """Set the pagination to F.[page].[a=recto/b=version] style.

        Args:
            recto_verso: Use recto-verso pagination if True. Defaults to True.

        Returns:
            The corpus loader object so that methods can be chained in the functional style.
        """

        self._recto_verso_pagination = recto_verso
        return self

    def _process_bag(self, bag, locators):
        # Prepares a bag, with or without locators as indicated. Adds the Tohoku numbers, applies folio segmentation,
        #   strips markup, removes spaces, and removes locators if requested.
        bag = bag \
            .map(_fn_to_tohs) \
            .flatten() \
            .map(lambda args: tuple(list(args) + [self.tag_treatments, self.ref_types_to_strip])) \
            .map(_process_fstr) \
            .flatten()
        if self._recto_verso_pagination:
            bag = bag \
                .map(_recto_verso_paginate)
        if not locators:
            bag = bag.map(lambda args: args[-1])
        return bag

    @property
    def dataframe(self):
        res_df = super().dataframe
        res_df["volume_number"] = res_df["volume_number"].fillna(
            res_df.filename.map(lambda x: x.split('-')[0])) \
            .astype(int)
        res_df = res_df[self._df_final_columns]
        return res_df
