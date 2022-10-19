import os
import pickle
import logging
from dataclasses import dataclass
from typing import List, Optional
from tqdm.auto import tqdm
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset as TorchDataset


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']

logger = logging.getLogger(__name__)


@dataclass
class TokenTagExample:
    """A single set of features for an example of a token classification dataset."""

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    labels: Optional[List[int]] = None


class TokenTagDataset(TorchDataset):
    """A PyTorch dataset for fine-tuning a Huggingface transformer model on a token classification dataset, such as the
        SOAS part-of-speech task.

    Args:
        processed_dataset: Name of the preprocessed dataset with a pickle file with tsheg-pretokenized tokens and
            labels. Will translate to the file $CAI_DATA_BASE_PATH/{processed_dataset}/dataset.pkl.
        concatenate_examples (bool): Concatenate examples to form long sentences. Defaults to False.
        use_mask_for_word_pieces (bool): When set to False, the first piece of a word in the dataset is marked with its
            label and the rest of the pieces of that word are marked with the padding token, which is set to the
            ignored index of the cross-entropy loss by default. When set to True the [MASK] token is used for this
            instead, which is not ignored by the loss, so that the model has to learn word segmentation and token
            classification end-to-end. Defaults to False.
        return_data_as_dict: Return dictionaries instead of dataclasses. Is expected by Transformers 4 data collator.
        max_seq_length: Maximum sequence length in the examples in the dataset. Defaults to 128.
        dupe_count: Number of duplicate walks through words in the SOAS dataset to make training examples. Meant to be
            used in combination with dupe_offset. The actual number of walks will be dupe_count + 1. Defaults to 0.
        dupe_offset: Offset when duplicating the walking through words in the SOAS dataset to make training examples.
            Defaults to 3.
    """

    _pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    _cls_token_segment_id = 0
    _sequence_segment_id = 0
    _pad_token_segment_id = 0

    processed_dataset = None
    concatenate_examples=False
    use_mask_for_word_pieces = False
    return_data_as_dict = True
    max_seq_length = 128
    dupe_count = 0
    dupe_offset = 3

    def __init__(self,
                 tokenizer,
                 processed_dataset,
                 verbose=False,
                 tqdm=tqdm,      # pylint: disable=redefined-outer-name
                 examples=None,
                 label_to_id_map=None):
        """Initialize the dataset. This constructor will prepare examples for training with the parameters specified in
            the class, especially max_seq_length.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for getting the token ids for the various special
                tokens like [CLS], [SEP] and <pad>.
            processed_dataset: Name of the preprocessed dataset with a pickle file with tsheg-pretokenized tokens and
                labels. Will translate to the file
                $CAI_DATA_BASE_PATH/{processed_dataset}/dataset.pkl.
            verbose (bool): If True will print a progress bar using TQDM. Defaults to False.
            tqdm (TQDM module, optional): Pass in the TQDM module to use for the progress bar. Useful when running in a
                notebook. Defaults to the tqdm you get when using "from tqdm.auto import tqdm".
            examples: A list of pre-loaded examples or None. If None then will load the dataset from the preprocessed
                tokens. Defaults to None.
            label_to_id_map: A mapping of part-of-speech text labels to their tag ids. Defaults to None which means to
                infer from the dataset. Pass in an extrnal mapping if you have two separate datasets that you want to
                agree on the labels, such as separately constructed test data.
        """

        super().__init__()
        if processed_dataset.endswith(".pkl"):
            self.processed_dataset = os.path.join(DATA_BASE_PATH, f"{processed_dataset}")
        else:
            self.processed_dataset = os.path.join(DATA_BASE_PATH, f"{processed_dataset}/dataset.pkl")
        self.tokenizer = tokenizer

        if examples is not None:
            self.examples = examples
            return

        if not os.path.isfile(self.processed_dataset):
            raise FileNotFoundError("Preprocessed data not found.")

        lock_path = os.path.join(
            os.environ['CAI_TEMP_PATH'],
            os.path.basename(self.processed_dataset.replace('/', '_').replace('\\', '_')) + ".lock")
        with FileLock(lock_path):
            with open(self.processed_dataset, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        concatted_data = isinstance(self.tokenized_data[0][1], list)
        examples_are_words = isinstance(self.tokenized_data[0][0][0], list)

        if concatted_data:
            unique_labels = sorted(list({label for example in self.tokenized_data for label in example[1]}))
        else:
            unique_labels = sorted(list({example[1] for example in self.tokenized_data}))
        if '[MASK]' in unique_labels:
            self.mask_token_id = unique_labels.index('[MASK]')
        else:
            self.mask_token_id = len(unique_labels) + 1
        if label_to_id_map is None:
            self.label_to_id_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.label_to_id_map['<pad>'] = TokenTagDataset._pad_token_label_id
            self.label_to_id_map['[MASK]'] = self.mask_token_id
        else:
            self.label_to_id_map = label_to_id_map
            for label in set(unique_labels).difference(set(self.label_to_id_map.keys())):
                self.label_to_id_map[label] = tokenizer.unk_token_id
        self.id_to_label_map = {idx: label for label, idx in self.label_to_id_map.items()}

        self.examples, skipped_examples = [], 0
        if not self.concatenate_examples:
            for cur_tokens, cur_labels in tqdm(self.tokenized_data, desc="Example", disable=not verbose):
                cur_labels = [self.label_to_id_map[l] for l in cur_labels]
                attention_mask = [1] * len(cur_tokens)
                token_type_ids = [self._sequence_segment_id] * len(cur_tokens)
                self.examples.append(
                    TokenTagExample(
                        input_ids=cur_tokens,
                        labels=cur_labels,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                )
        else:
            word_piece_id = self.mask_token_id if self.use_mask_for_word_pieces else self._pad_token_label_id
            input_ids, labels, attention_mask, token_type_ids = [tokenizer.bos_token_id], [word_piece_id], [1], \
                [self._cls_token_segment_id]
            special_tokens_count = tokenizer.num_special_tokens_to_add()

            for cur_dupe in tqdm(range(TokenTagDataset.dupe_count + 1),
                                desc="Duplicate",
                                disable=(not verbose) or (TokenTagDataset.dupe_count == 0)):
                offset = cur_dupe * TokenTagDataset.dupe_offset
                for cur_idx in tqdm(range(offset, len(self.tokenized_data)),
                                    desc="Example",
                                    disable=not verbose,
                                    leave=TokenTagDataset.dupe_count == 0):
                    cur_tokens, cur_labels = self.tokenized_data[cur_idx]
                    if examples_are_words and concatted_data:
                        new_tokens, new_labels = [], []
                        for token_batch, label in zip(cur_tokens, cur_labels):
                            new_tokens.extend(token_batch)
                            new_labels.extend([self.label_to_id_map[label]] + [word_piece_id] * (len(token_batch) - 1))
                        cur_tokens, cur_labels = new_tokens, new_labels
                    elif not concatted_data:
                        cur_labels = [self.label_to_id_map[cur_labels]] + [word_piece_id] * (len(cur_tokens) - 1)
                    if len(cur_tokens) > self.max_seq_length - special_tokens_count:
                        skipped_examples += 1
                        continue
                    new_len = len(input_ids) + len(cur_tokens)
                    if new_len >= self.max_seq_length - special_tokens_count:
                        padding_length = self.max_seq_length - special_tokens_count - len(input_ids) + 1
                        input_ids.extend([tokenizer.pad_token_id] * padding_length + [tokenizer.eos_token_id])
                        labels.extend([self._pad_token_label_id] * padding_length + [word_piece_id])
                        attention_mask.extend([0] * padding_length + [1])
                        token_type_ids.extend(
                            [self._pad_token_segment_id] * padding_length + [self._sequence_segment_id])

                        assert len(input_ids) <= self.max_seq_length
                        assert len(input_ids) == len(labels)
                        self.examples.append(
                            TokenTagExample(
                                input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids))
                        input_ids, labels, attention_mask, token_type_ids = [tokenizer.bos_token_id], \
                            [word_piece_id], [1], [self._cls_token_segment_id]

                    input_ids.extend(cur_tokens)
                    labels.extend(cur_labels)
                    attention_mask.extend([1] * len(cur_tokens))
                    token_type_ids.extend([self._sequence_segment_id] * len(cur_tokens))
        if skipped_examples > 0:
            logger.warning(f"Skipped {skipped_examples} examples")

    def __len__(self):
        # Return dataset length
        return len(self.examples)

    def __getitem__(self, i):
        # Return the i-th item from the dataset
        return self.examples[i].__dict__
