from copy import deepcopy

import torch
from tqdm.auto import tqdm


def opening_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return ['།' + sent if not sent[0] == '།' else sent for sent in bo_text.strip().split(' །') if len(sent) > 0]


def closing_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() + '།' for x in bo_text.strip().split('། ') if len(x.strip()) > 0]


def double_shad_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() for x in bo_text.strip().split('།།') if len(x.strip()) > 0]


def line_break_segmenter(bo_text, **kwargs):
    if not bo_text[0] == '།':
        bo_text = '།' + bo_text
    return [x.strip() for x in bo_text.split('\n') if len(x.strip()) > 0]


def target_token_count_segmenter(bo_text, tibert_tkn=None, max_register_length=128, num_special_tokens=2, **kwargs):
    if tibert_tkn is None:
        raise ValueError("Must pass in a TibertTokenizer")
    bo_segments = opening_shad_segmenter(bo_text)
    available_space = max_register_length - num_special_tokens
    bo_token_lengths = [len(tibert_tkn.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
    if max(bo_token_lengths) > available_space:
        new_segments = []
        for idx, (bo_segment, tkn_length) in enumerate(zip(bo_segments, bo_token_lengths)):
            if tkn_length > available_space:
                new_segments.extend(closing_shad_segmenter(bo_segment))
            else:
                new_segments.append(bo_segment)
        bo_segments = new_segments
    bo_token_lengths = [len(tibert_tkn.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
    if max(bo_token_lengths) > available_space:
        raise ValueError("Tokenized Tibetan text is too long for register encoding")
    bo_registers, register_start, register_idx = [], 0, 0
    while register_idx < len(bo_token_lengths):
        while sum(bo_token_lengths[register_start:register_idx + 1]) <= available_space:
            if register_idx == len(bo_token_lengths):
                break
            register_idx += 1
        if register_idx == register_start:
            continue
        bo_registers.append(' '.join(bo_segments[register_start:register_idx]).strip())
        register_start = register_idx
    return bo_registers


class TranslatorBase:
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument(
            "--num-beams",
            type=int,
            default=5,
            help="Number of beams in beam search")
        parser.add_argument(
            "--no-repeat-ngram-size",
            type=int,
            default=None,
            help="The length of the repeat n-gram stopper in beam search")
        parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=None,
            help="CTRL-style repetition penalty")
        parser.add_argument(
            "--repetition-penalty-slope",
            type=float,
            default=None,
            help="Slope for power term in the repetition penalty")
        parser.add_argument(
            "--repetition-penalty-window-length",
            type=int,
            default=None,
            help="Use this many last few tokens to calculate the repetition penalty")
        parser.add_argument(
            "--ngram-repetition-penalty",
            type=float,
            default=None,
            help="Penalty for repeating an n-gram, combine with no-repeat-ngram-size")

    def apply_args(self, parser_args):
        self.generator_kwargs['num_beams'] = parser_args.num_beams
        if parser_args.no_repeat_ngram_size is not None:
            self.generator_kwargs['no_repeat_ngram_size'] = parser_args.no_repeat_ngram_size
        if parser_args.repetition_penalty is not None:
            self.generator_kwargs['repetition_penalty'] = parser_args.repetition_penalty
        if parser_args.repetition_penalty_slope is not None:
            self.generator_kwargs['repetition_penalty_slope'] = parser_args.repetition_penalty_slope
        self.generator_kwargs['repetition_penalty_window_length'] = parser_args.repetition_penalty_window_length
        self.generator_kwargs['ngram_repetition_penalty'] = parser_args.ngram_repetition_penalty

    def __init__(self, tibert_tkn, tiebart, en_tokenizer):
        self.tibert_tkn = tibert_tkn
        self.tiebart = tiebart
        self.en_tokenizer = en_tokenizer
        self.segmenter_lambda = opening_shad_segmenter
        self.catch_translation_errors = True
        self.tqdm = tqdm
        self.reset_translation_state()

        self.generator_kwargs = {'num_beams': 5}

    def _safe_decode(self, gen_tkns):
        gen_tkns = [int(x) for x in gen_tkns.split()]
        if len(gen_tkns) == 0:
            return ''
        return self.en_tokenizer.decode(torch.LongTensor(gen_tkns))

    def _prep_models_for_translation(self):
        pass

    def _preprocess_sentences(self, sentences):
        return sentences

    def _translate_segment(self, bo_text, **kwargs):
        raise NotImplementedError()

    def reset_translation_state(self):
        pass

    def translate(self, bo_text, **kwargs):
        self._prep_models_for_translation()
        translations = []
        sentences = self.segmenter_lambda(bo_text)
        sentences = self._preprocess_sentences(sentences)
        for sentence in self.tqdm(sentences):
            if self.catch_translation_errors:
                try:
                    translation = self._translate_segment(sentence, **kwargs)
                except Exception:
                    translation = "ERROR"
            else:
                translation = self._translate_segment(sentence, **kwargs)
            translations.append(translation)
        return sentences, translations

    def __call__(self, bo_text, **kwargs):
        return self.translate(bo_text, **kwargs)


class TranslatorDirect(TranslatorBase):
    def _translate_segment(self, bo_text, num_beams=5):
        with torch.no_grad():
            encoded = torch.LongTensor([int(x) for x in self.tibert_tkn.encode(bo_text)])
            gen_tkns = self.tiebart.generate(encoded, **self.generator_kwargs)
            return self._safe_decode(' '.join(map(str, gen_tkns[0]['tokens'].tolist())))


class TranslatorRegisterDirect(TranslatorBase):
    def _preprocess_sentences(self, sentences):
        bo_registers = []
        for idx in range(0, len(sentences), self.tiebart.task.args.max_num_registers):
            bo_registers.append(sentences[idx:idx + self.tiebart.task.args.max_num_registers])
        return bo_registers

    def _translate_segment(self, bo_text, num_beams=5):
        if not self.tiebart.task.args.use_registers:
            raise ValueError("Task should have use_registers set to use this decoder. Did you load the right model?")
        with torch.no_grad():
            encoded_registers = [
                [int(x) for x in self.tibert_tkn.encode(register_text)] + [self.tiebart.task.args.eor_token_id]
                for register_text in bo_text]
            encoded_registers[-1] = encoded_registers[-1][:-1]
            encoded = torch.cat([torch.LongTensor(register) for register in encoded_registers])
            gen_tkns = self.tiebart.generate(encoded, **self.generator_kwargs)
            return self._safe_decode(' '.join(map(str, gen_tkns[0]['tokens'].tolist())))


class TranslatorAutoregressive(TranslatorBase):
    def reset_translation_state(self):
        super().reset_translation_state()
        self._incremental_states = None

    def _translate_segment(self, bo_text, num_beams=5):
        with torch.no_grad():
            encoded = torch.LongTensor([int(x) for x in self.tibert_tkn.encode(bo_text)])
            gen_tkns = self.tiebart.generate(
                encoded,
                inference_step_args={
                    'incremental_states': self._incremental_states},
                **self.generator_kwargs)
            tokens, self._incremental_states = gen_tkns[0]['tokens'], gen_tkns[0]['incremental_states']
            for model, incremental_state in zip(self.tiebart.models, self._incremental_states):
                for layer in model.decoder.layers:
                    del incremental_state[layer.encoder_attn._incremental_state_id + '.attn_state']
            return self._safe_decode(' '.join(map(str, tokens.tolist())))


class TranslatorWarmBeamSearch(TranslatorBase):
    def reset_translation_state(self):
        self._translation_state = ([], [])
        self.tiebart.task.warm_beam_search = True

    def _translate_segment(self, bo_text, num_beams=5):
        past_bo_tokens, past_beams = self._translation_state
        with torch.no_grad():
            cur_encoded = [int(x) for x in self.tibert_tkn.encode(bo_text, add_special_tokens=False)]

            max_seq_length = self.tiebart.models[0].encoder.tibert_mdl.config.embedding_size
            if len(cur_encoded) > max_seq_length - 2:
                raise ValueError("Line too long: " + bo_text)
            while sum([len(bo_tokens) for bo_tokens in past_bo_tokens]) > max_seq_length - len(cur_encoded) - 2:
                past_beams = past_beams[1:]
                past_bo_tokens = past_bo_tokens[1:]

            cur_beams = [[self.tiebart.task.target_dictionary.bos()] for _ in range(num_beams)]
            for past_beams_step in past_beams:
                for beam_idx in range(len(cur_beams)):
                    cur_beams[beam_idx].extend(past_beams_step[beam_idx])

            encoded = [self.tibert_tkn.bos_token_id]
            for bo_tokens in past_bo_tokens:
                encoded.extend(bo_tokens)
            encoded.extend(cur_encoded + [self.tibert_tkn.eos_token_id])

            self.tiebart.task.past_beams = cur_beams
            gen_tkns = self.tiebart.generate(torch.LongTensor(encoded), **self.generator_kwargs)
            gen_tkns = [gen_beam['tokens'].tolist()[1:-1] for gen_beam in gen_tkns]

            # Pick top beam only
            past_beams_len = sum([len(past_beam[0]) for past_beam in past_beams])
            gen_tkns = [gen_tkns[0][past_beams_len:] for _ in range(num_beams)]

            translation = self._safe_decode(' '.join(map(str, gen_tkns[0])))

            past_bo_tokens.append(cur_encoded)
            past_beams.append(gen_tkns)
            self._translation_state = (past_bo_tokens, past_beams)
            return translation


class TranslatorRegisterWarmBeamSearch(TranslatorBase):
    def reset_translation_state(self):
        self._translation_state = ([], [])
        self.tiebart.task.warm_beam_search = True

    def _translate_segment(self, bo_text, num_beams=5):
        if not self.tiebart.task.args.use_registers:
            raise ValueError("Task should have use_registers set to use this decoder. Did you load the right model?")

        past_bo_tokens, past_beams = self._translation_state
        with torch.no_grad():
            cur_encoded = [int(x) for x in self.tibert_tkn.encode(bo_text, add_special_tokens=False)]

            while len(past_bo_tokens) > self.tiebart.task.args.max_num_registers - 1:
                # Should actually only happen once during normal decoding
                past_beams = past_beams[1:]
                past_bo_tokens = past_bo_tokens[1:]

            cur_beams = [[self.tiebart.task.target_dictionary.bos()] for _ in range(num_beams)]
            for past_beams_step in past_beams:
                for beam_idx in range(len(cur_beams)):
                    cur_beams[beam_idx].extend(past_beams_step[beam_idx])

            bo_registers = deepcopy(past_bo_tokens)
            bo_registers.append(cur_encoded)

            encoded_registers = [
                [self.tibert_tkn.bos_token_id] + register_tokens +
                [self.tibert_tkn.eos_token_id, self.tiebart.task.args.eor_token_id]
                for register_tokens in bo_registers]
            encoded_registers[-1] = encoded_registers[-1][:-1]
            encoded = torch.cat([torch.LongTensor(register) for register in encoded_registers])

            self.tiebart.task.past_beams = cur_beams
            gen_tkns = self.tiebart.generate(encoded, **self.generator_kwargs)
            gen_tkns = [gen_beam['tokens'].tolist()[1:-1] for gen_beam in gen_tkns]

            # Pick top beam only
            past_beams_len = sum([len(past_beam[0]) for past_beam in past_beams])
            gen_tkns = [gen_tkns[0][past_beams_len:] for _ in range(num_beams)]

            translation = self._safe_decode(' '.join(map(str, gen_tkns[0])))

            past_bo_tokens.append(cur_encoded)
            past_beams.append(gen_tkns)
            self._translation_state = (past_bo_tokens, past_beams)
            return translation
