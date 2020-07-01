#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.data.legacy.masked_lm_dictionary import BertDictionary
from pytext import resources
from pytext.common.constants import Token
from pytext.config.component import ComponentType, create_component
from pytext.data.tensorizers import Tensorizer, TensorizerScriptImpl
from pytext.data.tokenizers import Tokenizer, WordPieceTokenizer
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK, Vocabulary
from pytext.torchscript.tensorizer.tensorizer import VocabLookup
from pytext.torchscript.utils import ScriptBatchInput, pad_2d, pad_2d_mask
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager
from pytext.utils.lazy import lazy_property


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, Token] = None,
    max_vocab: int = -1,
    min_count: int = -1,
    tokens_to_add: Optional[List[str]] = None,
) -> Vocabulary:
    """
    Function builds a PyText vocabulary for models pre-trained using Fairseq
    modules. The dictionary class can take any Fairseq Dictionary class
    and is used to load the vocab file.
    """
    dictionary = dictionary_class.load(vocab_file)
    # finalize will sort the dict based on frequency so only do this if
    # a min_count or max_vocab size is specified
    if min_count > 0 or max_vocab > 0:
        dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
    if tokens_to_add:
        for token in tokens_to_add:
            dictionary.add_symbol(token)
    return Vocabulary(
        dictionary.symbols, dictionary.count, replacements=special_token_replacements
    )


class BERTTensorizerBaseScriptImpl(TensorizerScriptImpl):
    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = ScriptVocabulary(
            list(vocab),
            pad_idx=vocab.get_pad_index(),
            bos_idx=vocab.get_bos_index(-1),
            eos_idx=vocab.get_eos_index(-1),
            unk_idx=vocab.get_unk_index(),
        )
        self.vocab_lookup = VocabLookup(self.vocab)
        self.max_seq_len = max_seq_len

    def _lookup_tokens(
        self, tokens: List[Tuple[str, int, int]], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        This function knows how to call lookup_tokens with the correct
        settings for this model. The default behavior is to wrap the
        numberized text with distinct BOS and EOS tokens. The resulting
        vector would look something like this:
        [BOS, token1_id, . . . tokenN_id, EOS]

        The function also takes an optional seq_len parameter which is
        used to customize truncation in case we have multiple text fields.
        By default max_seq_len is used. It's upto the numberize function of
        the class to decide how to use the seq_len param.

        For example:
        - In the case of sentence pair classification, we might want both
        pieces of text have the same length which is half of the
        max_seq_len supported by the model.
        - In the case of QA, we might want to truncate the context by a
        seq_len which is longer than what we use for the question.

        Args:
            tokens: a list of tokens represent a sentence, each token represented
            by token string, start and end indices.

        Returns:
            tokens_ids: List[int], a list of token ids represent a sentence.
            start_indices: List[int], each token start indice in the sentence.
            end_indices: List[int], each token end indice in the sentence.
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        return self.vocab_lookup(
            tokens,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=max_seq_len,
        )

    def _wrap_numberized_tokens(
        self, numberized_tokens: List[int], idx: int
    ) -> List[int]:
        """
        If a class has a non-standard way of generating the final numberized text
        (eg: BERT) then a class specific version of wrap_numberized_text function
        should be implemented. This allows us to share the numberize
        function across classes without having to copy paste code. The default
        implementation doesnt do anything.
        """
        return numberized_tokens

    def numberize(
        self, per_sentence_tokens: List[List[Tuple[str, int, int]]]
    ) -> Tuple[List[int], List[int], int, List[int]]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model.

        Args:
            per_sentence_tokens: list of tokens per sentence level in one row,
            each token represented by token string, start and end indices.

        Returns:
            tokens: List[int], a list of token ids, concatenate all
            sentences token ids.
            segment_labels: List[int], denotes each token belong to
            which sentence.
            seq_len: int, tokens length
            positions: List[int], token positions
        """
        tokens: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        positions: List[int] = []

        for idx, single_sentence_tokens in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(single_sentence_tokens)[0]
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)

            tokens.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))

        seq_len = len(tokens)
        positions = [i for i in range(seq_len)]
        return tokens, segment_labels, seq_len, positions

    def tensorize(
        self,
        tokens_2d: List[List[int]],
        segment_labels_2d: List[List[int]],
        seq_lens_1d: List[int],
        positions_2d: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert instance level vectors into batch level tensors.
        """
        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels = torch.tensor(
            pad_2d(segment_labels_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long
        )
        positions = torch.tensor(
            pad_2d(positions_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long
        )
        if self.device == "":
            return tokens, pad_mask, segment_labels, positions
        else:
            return (
                tokens.to(self.device),
                pad_mask.to(self.device),
                segment_labels.to(self.device),
                positions.to(self.device),
            )

    def tokenize(
        self,
        row_text: Optional[List[str]],
        row_pre_tokenized: Optional[List[List[str]]],
    ) -> List[List[Tuple[str, int, int]]]:
        """
        This function convert raw inputs into tokens, each token is represented
        by token(str), start and end indices in the raw inputs. There are two
        possible inputs to this function depends if the tokenized in implemented
        in TorchScript or not.

        Case 1: Tokenizer has a full TorchScript implementation, the input will
        be a list of sentences (in most case it is single sentence or a pair).

        Case 2: Tokenizer have partial or no TorchScript implementation, in most
        case, the tokenizer will be host in Yoda, the input will be a list of
        pre-processed tokens.

        Returns:
            per_sentence_tokens: tokens per setence level, each token is
            represented by token(str), start and end indices.
        """
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []

        if row_text is not None:
            for text in row_text:
                per_sentence_tokens.append(self.tokenizer.tokenize(text))
        elif row_pre_tokenized is not None:
            for sentence_pre_tokenized in row_pre_tokenized:
                sentence_tokens: List[Tuple[str, int, int]] = []
                for token in sentence_pre_tokenized:
                    sentence_tokens.extend(self.tokenizer.tokenize(token))
                per_sentence_tokens.append(sentence_tokens)

        return per_sentence_tokens

    def forward(
        self, inputs: ScriptBatchInput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wire up tokenize(), numberize() and tensorize() functions for data
        processing.
        When export to TorchScript, the wrapper module should choose to use
        texts or pre_tokenized based on the TorchScript tokenizer
        implementation (e.g use external tokenizer such as Yoda or not).
        """
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[int]] = []

        for idx in range(self.batch_size(inputs)):
            tokens: List[List[Tuple[str, int, int]]] = self.tokenize(
                self.get_texts_by_index(inputs.texts, idx),
                self.get_tokens_by_index(inputs.tokens, idx),
            )

            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(
                tokens
            )
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_lens_1d.append(numberized[2])
            positions_2d.append(numberized[3])

        return self.tensorize(tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d)

    def torchscriptify(self):
        # tokenizer will only be used in Inference, so we hold its torchscriptify
        # by end of the training.
        if not isinstance(self.tokenizer, torch.jit.ScriptModule):
            self.tokenizer = self.tokenizer.torchscriptify()
        return super().torchscriptify()


class BERTTensorizerBase(Tensorizer):
    """
    Base Tensorizer class for all BERT style models including XLM,
    RoBERTa and XLM-R.
    """

    __EXPANSIBLE__ = True

    class Config(Tensorizer.Config):
        # BERT style models support multiple text inputs
        columns: List[str] = ["text"]
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        # base token-level tokenizer for sequence labeling tasks
        base_tokenizer: Optional[Tokenizer.Config] = None
        vocab_file: str = ""
        max_seq_len: int = 256

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        base_tokenizer: Tokenizer = None,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.base_tokenizer = base_tokenizer
        self.max_seq_len = max_seq_len
        # Needed to ensure that we're not masking special tokens. By default
        # we use the BOS token from the vocab. If a class has different
        # behavior (eg: XLM), it needs to override this.
        self.bos_token = self.vocab.bos_token

    @property
    def column_schema(self):
        return [(column, str) for column in self.columns]

    @lazy_property
    def tensorizer_script_impl(self):
        return self.__TENSORIZER_SCRIPT_IMPL__(
            tokenizer=self.tokenizer, vocab=self.vocab, max_seq_len=self.max_seq_len
        )

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model.
        """
        per_sentence_tokens = [
            self.tokenizer.tokenize(row[column]) for column in self.columns
        ]
        return self.tensorizer_script_impl.numberize(per_sentence_tokens)

    def tensorize(self, batch) -> Tuple[torch.Tensor, ...]:
        """
        Convert instance level vectors into batch level tensors.
        """
        return self.tensorizer_script_impl.tensorize_wrapper(*zip(*batch))

    def initialize(self, vocab_builder=None, from_scratch=True):
        # vocab for BERT is already set
        return
        # we need yield here to make this function a generator
        yield

    def sort_key(self, row):
        return row[2]


class BERTTensorizerScriptImpl(BERTTensorizerBaseScriptImpl):
    def _lookup_tokens(
        self, tokens: List[Tuple[str, int, int]], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        return self.vocab_lookup(
            tokens,
            bos_idx=None,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=max_seq_len,
        )

    def _wrap_numberized_tokens(
        self, numberized_tokens: List[int], idx: int
    ) -> List[int]:
        if idx == 0:
            numberized_tokens = [self.vocab.bos_idx] + numberized_tokens
        return numberized_tokens


class BERTTensorizer(BERTTensorizerBase):
    """
    Tensorizer for BERT tasks.  Works for single sentence, sentence pair, triples etc.
    """

    __EXPANSIBLE__ = True
    __TENSORIZER_SCRIPT_IMPL__ = BERTTensorizerScriptImpl

    class Config(BERTTensorizerBase.Config):
        tokenizer: Tokenizer.Config = WordPieceTokenizer.Config()
        vocab_file: str = WordPieceTokenizer.Config().wordpiece_vocab_path

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        """
        from_config parses the config associated with the tensorizer and
        creates both the tokenizer and the Vocabulary object. The extra arguments
        passed as kwargs allow us to reuse thie function with variable number
        of arguments (eg: for classes which derive from this class).
        """
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        special_token_replacements = {
            "[UNK]": UNK,
            "[PAD]": PAD,
            "[CLS]": BOS,
            "[MASK]": MASK,
            "[SEP]": EOS,
        }
        if isinstance(tokenizer, WordPieceTokenizer):
            vocab = Vocabulary(
                [token for token, _ in tokenizer.vocab.items()],
                replacements=special_token_replacements,
            )
        else:
            config.vocab_file = (
                resources.roberta.RESOURCE_MAP[config.vocab_file]
                if config.vocab_file in resources.roberta.RESOURCE_MAP
                else config.vocab_file
            )
            with PathManager.open(config.vocab_file) as file_path:
                vocab = build_fairseq_vocab(
                    dictionary_class=BertDictionary,
                    vocab_file=file_path,
                    special_token_replacements=special_token_replacements,
                )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            **kwargs,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        **kwargs,
    ) -> None:
        super().__init__(
            columns=columns, vocab=vocab, tokenizer=tokenizer, max_seq_len=max_seq_len
        )
