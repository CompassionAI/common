import argparse
import colorama
from tqdm import tqdm

import os
import json
import math
import random
import unicodedata
from copy import deepcopy

from cai_common.data import ParallelTMXLoader, TeiLoader, KangyurLoader

DATA_BASE_PATH = os.environ['TIBERT_DATA_BASE_PATH']


def _shuffle_concatted_dataset(flat_data, args):
    import torch
    from transformers import BertTokenizer, BertForNextSentencePrediction

    if len(flat_data) == 0:
        return []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()
    model.cuda()
    batch_size = 2

    def score_valid_next(first_sent, second_sents):
        first_sent, second_sents = first_sent.lower(), [sent.lower() for sent in second_sents]
        encoding = tokenizer([first_sent] * len(second_sents), second_sents, return_tensors='pt', padding=True)
        encoding = {key: val.cuda() for key, val in encoding.items()}
        logits = model(**encoding)[0]
        softmax = torch.nn.functional.softmax(logits)
        return [x[0] for x in softmax.tolist()]

    remaining_data = deepcopy(flat_data)
    total_data_len = len(flat_data)
    flat_data = []
    cur_sent = remaining_data.pop(random.randrange(len(remaining_data)))
    num_fails = 0
    for _ in tqdm(range(math.floor(args.num_shuffled_elems_frac * total_data_len))):
        flat_data.append(cur_sent)
        found_next_sent = False
        for _ in range(0, args.shuffle_elem_find_tries, batch_size):
            try_batch_idxs = [random.randrange(len(remaining_data)) for _ in range(batch_size)]
            try_candidates = [remaining_data[idx]['english'] for idx in try_batch_idxs]
            scores = score_valid_next(cur_sent['english'], try_candidates)
            to_break = False
            for score, batch_idx in zip(scores, try_batch_idxs):
                candidate_idx = batch_idx
                if score > args.shuffle_elem_find_threshold:
                    to_break = True
                    break
            if to_break:
                found_next_sent = True
                break
        if not found_next_sent:
            num_fails += 1
        cur_sent = remaining_data.pop(candidate_idx)
    print(f"    Number of times failed to find next sentence: {num_fails}")
    return flat_data


def _pull_parallel_dataset(dask_client, args):
    # Loads flat training and test datasets of parallel sentences into memory from Dask
    print("Creating dataframe...", end=' ', flush=True)

    ParallelTMXLoader.data_glob = os.path.join(args.parallel_dataset_location, "*.tmx")
    if args.sort_by_starting_index:
        folio_df = TeiLoader(DATA_BASE_PATH, 'kangyur').dataframe
        folio_df['locator'] = folio_df \
            .tohoku_number \
            .str.lower() \
            .map(lambda x: x.replace('toh', '')) + '|' + folio_df.location.fillna('').str.lower()
        folio_df = folio_df.set_index('locator')

        parallel_df = ParallelTMXLoader(DATA_BASE_PATH) \
            .apply_markup() \
            .clean_bad_chars() \
            .dataframe
        parallel_df['locator'] = parallel_df.tohoku.str.lower() + '|' + parallel_df.folio.str.lower()

        joined_df = parallel_df.join(folio_df, on='locator', rsuffix="_folio", how='outer')
        joined_df = joined_df[['tohoku', 'volume_number', 'location', 'tibetan', 'english', 'text']]
        joined_df['volume_number'] = joined_df.volume_number.fillna(-1).astype('int64')
        joined_df['text'] = joined_df.text.fillna('')
        joined_df['english'] = joined_df.english.fillna('')
        joined_df['start_idx'] = joined_df.apply(
            lambda row: ' '.join(row.text.split()).find(' '.join(row.english.split())), axis=1, meta=(None, 'int64'))
        joined_df = joined_df[joined_df.start_idx >= 0]
        txm_df = dask_client.persist(joined_df)[["tohoku", "tibetan", "english", "location", "start_idx"]]
    else:
        txm_df = ParallelTMXLoader(DATA_BASE_PATH) \
            .apply_markup() \
            .clean_bad_chars() \
            .dataframe
        txm_df = dask_client.persist(txm_df)[["tohoku", "tibetan", "english"]]

    print("Loading training dataframe...", end=' ', flush=True)
    train_df = txm_df[~txm_df.tohoku.isin(args.test_tohoku_nums)].compute()
    if args.sort_by_starting_index:
        train_df = train_df.sort_values(["tohoku", "location", "start_idx"])[["tibetan", "english"]].dropna()
    train_df = train_df[["tibetan", "english"]]
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)
    print("Loading test dataframe...", end=' ', flush=True)
    test_df = txm_df[txm_df.tohoku.isin(args.test_tohoku_nums)].compute()
    if args.sort_by_starting_index:
        test_df = test_df.sort_values(["tohoku", "location", "start_idx"])[["tibetan", "english"]].dropna()
    test_df = test_df[["tibetan", "english"]]
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("    Number of Tibetan characters in test data:", end=' ', flush=True)
    print(int(test_df.tibetan.map(len).sum()))
    print("    Number of sentences in test data:", end=' ', flush=True)
    print(int(test_df.tibetan.count()))

    train_flat_data, test_flat_data = train_df.to_dict(orient="records"), test_df.to_dict(orient="records")

    if args.shuffle_concats:
        shuffled_train_data, shuffled_test_data = [], []
        for _ in range(args.num_shuffling_repetitions):
            shuffled_train_data.extend(_shuffle_concatted_dataset(train_flat_data, args))
            shuffled_test_data.extend(_shuffle_concatted_dataset(test_flat_data, args))
        train_flat_data, test_flat_data = shuffled_train_data, shuffled_test_data

    return train_flat_data, test_flat_data


def _pull_folio_dataset(dask_client, args):
    # Loads flat training and test datasets of parallel folios into memory from Dask
    english_df = TeiLoader(DATA_BASE_PATH, "kangyur").dataframe
    kangyur_df = KangyurLoader(DATA_BASE_PATH).dataframe

    kangyur_df['locator'] = kangyur_df.apply(lambda row: str(row.volume_number) + '|' + str(row.location), axis=1)
    english_df['locator'] = english_df.apply(lambda row: str(row.volume_number) + '|' + str(row.location), axis=1)
    kangyur_df, english_df = kangyur_df.set_index('locator'), english_df.set_index('locator')
    joined_df = kangyur_df[['filename', 'text']].join(
        english_df[['filename', 'volume_number', 'tohoku_number', 'location', 'text']],
        how='inner',
        lsuffix="_tibetan")

    local_df = joined_df.compute()
    local_df = local_df[~local_df.index.duplicated(keep='first')] \
        .rename(columns={
            "text_tibetan": "tibetan",
            "text": "english"})[["tibetan", "english", "tohoku_number"]]

    test_tohoku_nums = ['toh' + str(num) if not num[0] == 't' else str(num) for num in args.test_tohoku_nums]
    train_df = local_df[~local_df.tohoku_number.isin(test_tohoku_nums)]
    test_df = local_df[local_df.tohoku_number.isin(test_tohoku_nums)]

    train_flat_data = train_df[["tibetan", "english"]].to_dict(orient="records")
    test_flat_data = test_df[["tibetan", "english"]].to_dict(orient="records")
    return train_flat_data, test_flat_data


def _pull_dictionary_dataset(dask_client, args):
    # Loads flat training dataset of dictionary words into memory from Dask. The test dataset is always empty.
    #   Optionally also applies simple length-based heuristics to only pick out well-defined words without long
    #   dictionary entries, and picks out the shortest definition from a comma-delimited list of definitions.
    from tibert.dict import TibetanDict, TibetanEncoding
    flat_data = []
    dict_ = TibetanDict(
        None,
        glob_override=args.dictionary_augment_glob.strip("\""),
        default_encoding=TibetanEncoding.UNICODE)
    for bo, ens in dict_.items():
        if not bo[-1] == '་':
            bo += '་'
        if args.pick_best_word:
            en_lengths = max(map(len, ens))
            if en_lengths < args.well_defined_word_max_en_len:
                ens = [en_split.strip() for en in ens for en_split in en.split(',')]
                en_lengths = list(map(len, ens))
                flat_data.append({
                    "tibetan": bo,
                    "english": ens[en_lengths.index(min(en_lengths))]})
        else:
            for en in ens:
                flat_data.append({
                    "tibetan": bo,
                    "english": en})
    return flat_data, []


def _prep_linear_dataset(flat_data, args):
    # No preprocessing, direct passthrough of dataset
    return flat_data


def _prep_concatted_dataset(flat_data, args):
    # Prepare a dataset where consecutive sentences are concatenated to form longer training examples
    from tibert.models import TibertTokenizer

    tibert_tkn = TibertTokenizer.from_pretrained('tibert-bpe-large')
    tibert_tkn.stochastic_tokenization = False
    bo_token_lengths = [
        len(tibert_tkn.encode(datum['tibetan'], add_special_tokens=False)) for datum in tqdm(flat_data)]

    concat_window = args.concat_window
    concatted_data = []
    for i, datum in tqdm(enumerate(flat_data), total=len(flat_data)):
        cur_datum = {
            "tibetan": "",
            "english": ""}
        bo_token_count = 2
        for j in range(concat_window):
            if i + j < len(flat_data):
                bo_token_count += bo_token_lengths[i + j]
                if bo_token_count > args.max_source_length:
                    continue
                cur_datum = deepcopy(cur_datum)
                cur_datum['tibetan'] = (cur_datum['tibetan'] + ' ' + flat_data[i + j]['tibetan']).strip()
                cur_datum['english'] = (cur_datum['english'] + ' ' + flat_data[i + j]['english']).strip()
                concatted_data.append(cur_datum)
    return concatted_data


def _prep_concatted_register_dataset(flat_data, args):
    # Prepare a dataset where consecutive sentences are concatenated to form longer training examples and split into
    #   source language registers of a given maximum length
    import torch
    from tibert.models import TibertTokenizer

    tibert_tkn = TibertTokenizer.from_pretrained('tibert-bpe-large')
    tibert_tkn.stochastic_tokenization = False
    bart_base = torch.hub.load('pytorch/fairseq', 'bart.base')
    bo_token_lengths = [
        len(tibert_tkn.encode(datum['tibetan'], add_special_tokens=False)) for datum in tqdm(flat_data)]
    en_token_lengths = [bart_base.encode(datum['english']).size()[0] - 2 for datum in tqdm(flat_data)]

    concatted_data = []
    for i, datum in tqdm(enumerate(flat_data), total=len(flat_data)):
        # First, append the greedy segmentation
        bo_registers, bo_length, en_length, en_line = [], 0, 0, ""
        cur_idx = i
        for _ in range(args.max_num_registers):
            bo_register, register_start = "", cur_idx
            while sum(bo_token_lengths[register_start:cur_idx + 1]) <= args.max_register_length - 2:
                if cur_idx >= len(flat_data) or en_length + en_token_lengths[cur_idx] >= args.max_target_length - 2:
                    break
                bo_register += ' ' + flat_data[cur_idx]['tibetan']
                en_line += ' ' + flat_data[cur_idx]['english']
                bo_length += bo_token_lengths[cur_idx]
                en_length += en_token_lengths[cur_idx]
                cur_idx += 1
            bo_register = bo_register.strip()
            if len(bo_register) > 0:
                bo_registers.append(bo_register)
        if bo_length == 0:
            continue
        concatted_data.append({
            'tibetan': bo_registers,
            'english': en_line.strip()})
        # Second, generate intermediate segmentations
        while True:
            if random.random() > args.intermediate_segmentation_probability:
                break
            bo_registers, en_line, en_length = [], "", 0
            cur_idx = i
            for _ in range(args.max_num_registers):
                top_idx, top_en_length = cur_idx, en_length
                while sum(bo_token_lengths[cur_idx:top_idx + 1]) <= args.max_register_length - 2:
                    if top_idx >= len(flat_data) or \
                       top_en_length + en_token_lengths[top_idx] >= args.max_target_length - 2:
                        break
                    top_en_length += en_token_lengths[top_idx]
                    top_idx += 1
                if top_idx == cur_idx:
                    break
                break_point = math.ceil(random.random() * (top_idx - cur_idx))
                break_data = flat_data[cur_idx:cur_idx + break_point]
                bo_register = ' '.join([datum['tibetan'] for datum in break_data]).strip()
                if len(bo_register) > 0:
                    bo_registers.append(bo_register)
                en_line += ' ' + ' '.join([datum['english'] for datum in break_data])
                en_length += sum(en_token_lengths[cur_idx:cur_idx + break_point])
                cur_idx += break_point
            concatted_data.append({
                'tibetan': bo_registers,
                'english': en_line.strip()})
    return concatted_data


def _prep_folio_register_dataset(flat_data, args):
    # Prepare a dataset of folios that have been split into registers
    from tibert.models import TibertTokenizer
    from tibert.utils import closing_shad_segmenter

    tibert_tkn = TibertTokenizer.from_pretrained('tibert-bpe-large')
    tibert_tkn.stochastic_tokenization = False

    concatted_data = []
    for i, datum in tqdm(enumerate(flat_data), total=len(flat_data)):
        # First, append the greedy segmentation
        if len(datum['tibetan']) == 0 or len(datum['english']) == 0:
            continue
        bo_segments = closing_shad_segmenter(datum['tibetan'])
        bo_token_lengths = [len(tibert_tkn.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
        bo_registers, register_start, register_idx = [], 0, 0
        for _ in range(args.max_num_registers):
            while sum(bo_token_lengths[register_start:register_idx + 1]) <= args.max_register_length - 2:
                if register_idx == len(bo_token_lengths):
                    break
                register_idx += 1
            if register_idx == register_start:
                continue
            bo_registers.append(' '.join(bo_segments[register_start:register_idx]).strip())
            register_start = register_idx
        if register_idx < len(bo_token_lengths):
            continue
        concatted_data.append({
            'tibetan': bo_registers,
            'english': datum['english']})
        # Second, generate intermediate segmentations
        while True:
            if random.random() > args.intermediate_segmentation_probability:
                break
            for num_tries in range(args.num_intermediate_tries):
                bo_registers, register_start, register_idx = [], 0, 0
                for _ in range(args.max_num_registers - 1):
                    while sum(bo_token_lengths[register_start:register_idx + 1]) <= args.max_register_length - 2:
                        if register_idx == len(bo_token_lengths):
                            break
                        register_idx += 1
                    if register_idx == register_start:
                        continue
                    break_point = math.ceil((register_idx - register_start) * random.random()) + register_start
                    bo_registers.append(' '.join(bo_segments[register_start:break_point]).strip())
                    register_start = break_point
                if sum(bo_token_lengths[register_start:]) > args.max_register_length - 2:
                    continue
                bo_registers.append(' '.join(bo_segments[register_start:]).strip())
                concatted_data.append({
                    'tibetan': bo_registers,
                    'english': datum['english']})
                break
            if num_tries == args.num_intermediate_tries:
                print("Intermediate segmentation failed for a folio!")

    return concatted_data


def _write_to_file(f_name, lines, cleaned_symbols, preprocess_location, separator=None):
    # Save dataset lines to a file while cleaning out bad symbols
    print(f_name, flush=True, end=' ')
    if separator is not None:
        separator = separator.strip() + ' '
    preprocess_location = os.path.join(DATA_BASE_PATH, preprocess_location)
    with open(os.path.join(preprocess_location, f_name), mode="w", encoding="utf-8") as f:
        for line in lines:
            if type(line) is list:
                if separator is None:
                    raise ValueError("Dataset requires a separator but none provided")
                line = separator.join(line)
            for bad_c, good_c in cleaned_symbols.items():
                line = line.replace(bad_c, good_c)
            f.write(line + '\n')


def _check_for_unks(f_name, en_lm, preprocess_location):
    # Check if there are any tokens in a file that encode into <unk>
    preprocess_location = os.path.join(DATA_BASE_PATH, preprocess_location)

    decoded, encoded = [], []
    with open(os.path.join(preprocess_location, f_name), mode="r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc=f_name):
            decoded.append(line)
            encoded.append(en_lm.encode(line))

    unk_id, unk_lines = en_lm.task.target_dictionary.unk(), []
    for line, line_text in zip(encoded, decoded):
        if unk_id in line.tolist():
            unk_lines.append((line, line_text))

    if len(unk_lines) > 0:
        print(f"Unknown tokens found in {f_name}!!!")
        for line in unk_lines[:10]:
            print("    " + line)
        if len(unk_lines) > 9:
            print("...")


def _postprocess_training_data(flat_data):
    if len(flat_data) == 0:
        return []
    if type(flat_data[0]['tibetan']) is list:
        return [
            {
                "tibetan": [subdatum.strip() for subdatum in datum["tibetan"]],
                "english": datum["english"].strip()}
            for datum in flat_data]
    else:
        return [
            {
                "tibetan": datum["tibetan"].strip(),
                "english": datum["english"].strip()}
            for datum in flat_data]


def _postprocess_final_data(bo_data, en_data, args):
    def _postprocess(text):
        text = text.replace(" ། ", " །")
        if text[-2:] == " །":
            text = text[:-2]
        return text

    if len(bo_data) == 0:
        return [], []
    if type(bo_data[0]) is list:
        bo_data = [[_postprocess(subdatum) for subdatum in datum] for datum in bo_data]
    else:
        bo_data = [_postprocess(datum) for datum in bo_data]

    if hasattr(args, "max_register_length"):
        from tibert.models import TibertTokenizer

        prev_len = len(bo_data)

        tibert_tkn = TibertTokenizer.from_pretrained('tibert-bpe-large')
        tibert_tkn.stochastic_tokenization = False
        if type(bo_data[0]) is list:
            bo_token_lengths = [
                max([len(tibert_tkn.encode(subdatum)) for subdatum in datum]) for datum in tqdm(bo_data)]
            bo_data = [
                datums for datums, len_ in zip(bo_data, bo_token_lengths) if len_ < args.max_register_length]
            en_data = [
                datums for datums, len_ in zip(en_data, bo_token_lengths) if len_ < args.max_register_length]
        else:
            bo_token_lengths = [len(tibert_tkn.encode(datum)) for datum in tqdm(bo_data)]
            bo_data = [
                datum for datum, len_ in zip(bo_data, bo_token_lengths) if len_ < args.max_register_length]
            en_data = [
                datum for datum, len_ in zip(en_data, bo_token_lengths) if len_ < args.max_register_length]

        print(f"Final data keeps {len(bo_data) / prev_len} of the original")

    return bo_data, en_data


def _check_equal_lengths(bo_file_name, en_file_name, preprocess_location):
    # Check if the given processed files are equal length
    preprocess_location = os.path.join(DATA_BASE_PATH, preprocess_location)
    bo_len = len(open(os.path.join(preprocess_location, bo_file_name), mode="r", encoding="utf-8").readlines())
    en_len = len(open(os.path.join(preprocess_location, en_file_name), mode="r", encoding="utf-8").readlines())
    assert bo_len == en_len


def _remove_accents(en_str):
    # Remove accents from an English string. First normalizes to NFKD and then removes all combining characters.
    nfkd_form = unicodedata.normalize('NFKD', en_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _prep_parser_shuffling(parser):
    parser.add_argument(
        "--shuffle-concats",
        help="Shuffle the sentences before concatenating",
        action="store_true")
    parser.add_argument(
        "--num-shuffling-repetitions",
        type=int,
        default=1,
        help="Number of times to repeat the shuffle to get a larger dataset")
    parser.add_argument(
        "--num-shuffled-elems-frac",
        type=float,
        default=0.5,
        help="Number of shuffled elements to get from the flat data")
    parser.add_argument(
        "--shuffle-elem-find-tries",
        type=int,
        default=100,
        help="Number of times to try to find the next sentences during shuffling")
    parser.add_argument(
        "--shuffle-elem-find-threshold",
        type=float,
        default=0.999,
        help="Score threshold to accept the next sentence")


def _prep_parser_registers(parser):
    parser.add_argument(
        '--max-num-registers',
        type=int,
        default=8,
        help='Maximum number of source language registers')
    parser.add_argument(
        '--max-register-length',
        type=int,
        default=128,
        help='Maximum number of tokens accepted by a register (including bos and eos)')
    parser.add_argument(
        '--intermediate-segmentation-probability',
        type=float,
        default=0.25,
        help='Probability of (fraction of) intermediate, as opposed to greedy, segmentations')


def _prep_parser(parser):
    # Prepare arg parser
    parser.add_argument(
        "--no-dask-distributed",
        help="Don't use the Dask distributed scheduler. Use this in case of strange errors. Will not make a "
             "dashboard",
        action="store_true")
    parser.add_argument(
        "--dask-workers",
        type=int,
        help="Number of Dask worker processes to load training data with. Defaults to 10",
        default=10)
    parser.add_argument(
        "--local-dask-dash",
        help="Make the Dask dashboard only visible on localhost. Use if getting weird crashes involving LocalCluster",
        action="store_true")
    parser.add_argument(
        "--parallel-dataset-location",
        type=str,
        help="Location of the parallel sentences dataset, under TIBERT_DATA_BASE_PATH. Defaults to "
             "'84000/data-translation-memory'",
        default="84000/data-translation-memory")
    parser.add_argument(
        "--sort-by-starting-index",
        action="store_true",
        help="Sort the parallel dataset by index of the English sentence in the parallel folio dataset")
    parser.add_argument(
        "--preprocess-location",
        type=str,
        help="Location where to save the preprocessed dataset, under TIBERT_DATA_BASE_PATH. Defaults to enbo_data/",
        default="enbo_data/")
    parser.add_argument(
        '--symbol-cleaning-json',
        type=str,
        default="symbol_cleaning.json",
        help='Name of file with symbol clean-up mapping. Defaults to symbol_cleaning.json')
    parser.add_argument(
        '--validation-frac',
        type=float,
        default=0,
        help='Fraction of training data to reserve for validation')
    parser.add_argument(
        '--test-tohoku-nums',
        nargs='*',
        type=str,
        help='Tohoku numbers of texts to use as test data')
    parser.add_argument(
        '--skip-check-for-en-unks',
        help='Skip the check of the English data for unknown tokens. Defaults to False',
        action="store_true")
    parser.add_argument(
        '--seed',
        default=12345,
        type=int,
        help='Seed for random numbers')
    parser.add_argument(
        '--separator',
        default=None,
        type=str,
        help='Separator for datasets with multiple registers')
    parser.add_argument(
        "--lower-case-en",
        help="Lower case the English translations",
        action="store_true")
    parser.add_argument(
        "--remove-en-accents",
        help="Remove accents from the English translations",
        action="store_true")

    subparsers = parser.add_subparsers(help='Different methods for preparing the parallel dataset.')

    naive_concats_parser = subparsers.add_parser("naive_concats")
    naive_concats_parser.add_argument(
        '--concat-window',
        type=int,
        help='Concatenate up to this many consecutive sentences to form longer training examples')
    naive_concats_parser.add_argument(
        '--max-source-length',
        type=int,
        default=128,
        help='Maximum length of the source model, including special tokens')
    _prep_parser_shuffling(naive_concats_parser)
    naive_concats_parser.set_defaults(
        pull_func=_pull_parallel_dataset,
        prep_func=_prep_concatted_dataset)

    concatted_registers_parser = subparsers.add_parser("concatted_registers")
    concatted_registers_parser.add_argument(
        '--max-target-length',
        type=int,
        default=1024,
        help='Maximum number of tokens output by the decoder (including bos and eos)')
    _prep_parser_shuffling(concatted_registers_parser)
    _prep_parser_registers(concatted_registers_parser)
    concatted_registers_parser.set_defaults(
        pull_func=_pull_parallel_dataset,
        prep_func=_prep_concatted_register_dataset)

    folio_registers_parser = subparsers.add_parser("folio_registers")
    folio_registers_parser.add_argument(
        '--num-intermediate-tries',
        type=int,
        default=10,
        help='Number of times to try to make an intermediate segmentation')
    _prep_parser_registers(folio_registers_parser)
    folio_registers_parser.set_defaults(
        pull_func=_pull_folio_dataset,
        prep_func=_prep_folio_register_dataset)

    dictionary_parser = subparsers.add_parser("dictionary")
    dictionary_parser.add_argument(
        '--dictionary-augment-glob',
        default=None,
        type=str,
        help='Glob for dictionaries to augment with. Defaults to None')
    dictionary_parser.add_argument(
        '--pick-best-word',
        help='Pick out the best word. If not set, will simply flatten the dictionaries',
        action="store_true")
    dictionary_parser.add_argument(
        '--well-defined-word-max-en-len',
        default=20,
        type=int,
        help='Maximum length of every English definition for a word to count as well-defined')
    dictionary_parser.set_defaults(
        pull_func=_pull_dictionary_dataset,
        prep_func=_prep_linear_dataset)


def main(args):
    if args.test_tohoku_nums is None:
        args.test_tohoku_nums = []

    random.seed(args.seed)

    print("Spinning up Dask cluster...", end=' ', flush=True)
    from dask.distributed import Client, LocalCluster
    dask_client = Client(LocalCluster(
        n_workers=args.dask_workers,
        threads_per_worker=1,
        ip="localhost" if args.local_dask_dash else "*"))
    print(
        colorama.Fore.GREEN +
        "Dashboard is at " + colorama.Style.BRIGHT + dask_client.dashboard_link + colorama.Style.RESET_ALL)

    train_flat_data, test_flat_data = args.pull_func(dask_client, args)
    train_flat_data = _postprocess_training_data(train_flat_data)
    test_flat_data = _postprocess_training_data(test_flat_data)

    print("Preparing datasets in memory...", flush=True, end=' ')
    prep_func = args.prep_func
    train_concat_data = prep_func(train_flat_data, args)
    test_concat_data = prep_func(test_flat_data, args)

    train_bo, valid_bo, train_en, valid_en = [], [], [], []
    for datum in train_concat_data:
        line_bo, line_en = datum['tibetan'], datum['english']
        cur_rand = random.random()
        if cur_rand < args.validation_frac:
            valid_bo.append(line_bo)
            valid_en.append(line_en)
            continue
        train_bo.append(line_bo)
        train_en.append(line_en)
    test_bo, test_en = [], []
    for datum in test_concat_data:
        line_bo, line_en = datum['tibetan'], datum['english']
        test_bo.append(line_bo)
        test_en.append(line_en)
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("Deduping...", flush=True, end=' ')
    split_by_pipe = type(train_bo[0]) is list
    if split_by_pipe:
        train_bo = ['|'.join(bo_segments) for bo_segments in train_bo]
    deduped = list(set(zip(train_bo, train_en)))
    train_bo, train_en = [bo for bo, _ in deduped], [en for _, en in deduped]
    if split_by_pipe:
        train_bo = [bo_segments.split('|') for bo_segments in train_bo]
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("Post-processing Tibetan dataset...", flush=True)
    train_bo, train_en = _postprocess_final_data(train_bo, train_en, args)
    valid_bo, valid_en = _postprocess_final_data(valid_bo, valid_en, args)
    test_bo, test_en = _postprocess_final_data(test_bo, test_en, args)
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("Post-processing English dataset...", flush=True, end=' ')
    if args.lower_case_en:
        train_en, valid_en, test_en = [[en.lower() for en in en_set] for en_set in [train_en, valid_en, test_en]]
    if args.remove_en_accents:
        train_en, valid_en, test_en = [[_remove_accents(en) for en in en_set]
                                       for en_set in [train_en, valid_en, test_en]]
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("Writing datasets to disk...", flush=True, end=' ')
    with open(args.symbol_cleaning_json, 'r') as f:
        cleaned_symbols_en = json.load(f)
    cleaned_symbols_bo = {"\n": ""}
    _write_to_file("train.bo", train_bo, cleaned_symbols_bo, args.preprocess_location, separator=args.separator)
    _write_to_file("train.en", train_en, cleaned_symbols_en, args.preprocess_location)
    _write_to_file("valid.bo", valid_bo, cleaned_symbols_bo, args.preprocess_location, separator=args.separator)
    _write_to_file("valid.en", valid_en, cleaned_symbols_en, args.preprocess_location)
    _write_to_file("test.bo", test_bo, cleaned_symbols_bo, args.preprocess_location, separator=args.separator)
    _write_to_file("test.en", test_en, cleaned_symbols_en, args.preprocess_location)
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    print("Checking for equal lengths...", flush=True, end=' ')
    print("train...", flush=True, end=' ')
    _check_equal_lengths("train.bo", "train.en", args.preprocess_location)
    print("valid...", flush=True, end=' ')
    _check_equal_lengths("valid.bo", "valid.en", args.preprocess_location)
    print("test...", flush=True, end=' ')
    _check_equal_lengths("test.bo", "test.en", args.preprocess_location)
    print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)

    if not args.skip_check_for_en_unks:
        print("Checking for unknown tokens...", flush=True)
        import torch
        en_lm = torch.hub.load("pytorch/fairseq", "bart.base")
        _check_for_unks("train.en", en_lm, args.preprocess_location)
        _check_for_unks("valid.en", en_lm, args.preprocess_location)
        _check_for_unks("test.en", en_lm, args.preprocess_location)
        print(colorama.Fore.GREEN + 'DONE' + colorama.Fore.RESET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _prep_parser(parser)
    args = parser.parse_args()

    main(args)
