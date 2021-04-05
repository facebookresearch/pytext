#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Type

from . import Batcher, Data
from .data import RowData
from .sources import DataSource
from .tensorizers import Tensorizer
from .utils import Vocabulary


def get_decoupled(tokens, filter_ood_slots):
    """
    Convert the seqlogical form to the decoupled form
    """

    def is_intent_token(token):
        return token.lower().startswith("[in:") or is_ood_token(token)

    def is_slot_token(token):
        return token.lower().startswith("[sl:")

    def is_ood_token(token):
        return token.lower().startswith("[") and token.lower().endswith("outofdomain")

    if not tokens:
        return []
    if filter_ood_slots:
        if is_ood_token(tokens[0]):
            # Out of domain sample
            return [tokens[0], "]"]

    decoupled = []
    # stack contains mutable tuples of
    # [index_of_opening_bracket: int, has_children: bool]
    stack = []
    for i, token in enumerate(tokens):
        if is_intent_token(token) or is_slot_token(token):
            # everything on stack now has children
            for tup in stack:
                tup[1] = True
            # add an opening bracket to stack
            stack.append([i, False])
            # add to decoupled form
            decoupled.append(tokens[i])
        elif token == "]":
            # no bracket to end
            if len(stack) == 0:
                raise ValueError(" ".join(tokens))
            idx, has_child = stack.pop()
            # don't keep tokens if it is an intent OR it has children
            if has_child or is_intent_token(tokens[idx]):
                decoupled.append(token)
            else:
                # leaf level slot: keep all tokens
                decoupled.extend(tokens[idx + 1 : i + 1])
        else:
            # normal token outside of a bracket
            if len(stack) == 0:
                raise ValueError(" ".join(tokens))
    if len(stack) > 0:
        raise ValueError(" ".join(tokens))
    return decoupled


def get_blind_decoupled(bracket_form, filter_ood_slots=True, **kwargs):
    return " ".join(get_decoupled(bracket_form.split(), filter_ood_slots))


class DecoupledSeq2SeqData(Data):
    class Config(Data.Config):
        sort_key: str = "src_seq_tokens"
        # Whether source/target need to be converted from seqlogical to decoupled form.
        decoupled_source: bool = False
        decoupled_target: bool = False
        # As we are interfacing with external libraries, we need to replace PyText's
        # custom type for special tokens with simple strings. The tensorizers' vocab
        # will be overridden to make use of the special tokens specified below.
        unk_token: str = "<unk>"
        pad_token: str = "<pad>"
        bos_token: str = "<bos>"
        eos_token: str = "<eos>"
        mask_token: str = "<mask>"
        # Whether we should be removing all slots if top level domain is OOD
        filter_target_ood_slots: bool = True
        # For cloze-style parsing, ontology tokens appear in the source sequence, thus
        # this controls whether the source tensorizer will receive the merged vocab.
        merge_source_vocab: bool = False

    @classmethod
    def from_config(
        cls,
        config: Config,
        schema: Dict[str, Type],
        tensorizers: Dict[str, Tensorizer],
        rank=0,
        world_size=1,
        init_tensorizers=True,
        **kwargs,
    ):
        return super().from_config(
            config,
            schema,
            tensorizers,
            rank,
            world_size,
            init_tensorizers,
            decoupled_source=config.decoupled_source,
            decoupled_target=config.decoupled_target,
            filter_target_ood_slots=config.filter_target_ood_slots,
            merge_source_vocab=config.merge_source_vocab,
            unk_token=config.unk_token,
            pad_token=config.pad_token,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
            mask_token=config.mask_token,
            **kwargs,
        )

    @staticmethod
    def _replace_tokens(vocab, unk_token, pad_token, bos_token, eos_token, mask_token):
        replacements = {
            vocab.unk_token: unk_token,
            vocab.pad_token: pad_token,
            vocab.bos_token: bos_token,
            vocab.eos_token: eos_token,
            vocab.mask_token: mask_token,
        }
        vocab.replace_tokens(
            dict(filter(lambda x: x[0] in vocab, replacements.items()))
        )
        vocab.unk_token = unk_token
        vocab.pad_token = pad_token
        vocab.bos_token = bos_token
        vocab.eos_token = eos_token
        vocab.mask_token = mask_token

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
        sort_key: Optional[str] = None,
        in_memory: Optional[bool] = False,
        init_tensorizers: Optional[bool] = True,
        init_tensorizers_from_scratch: Optional[bool] = True,
        decoupled_source: bool = False,
        decoupled_target: bool = False,
        filter_target_ood_slots: bool = True,
        merge_source_vocab: bool = False,
        unk_token: str = Config.unk_token,
        pad_token: str = Config.pad_token,
        bos_token: str = Config.bos_token,
        eos_token: str = Config.eos_token,
        mask_token: str = Config.mask_token,
    ):
        super().__init__(
            data_source,
            tensorizers,
            batcher,
            sort_key,
            in_memory,
            init_tensorizers,
            init_tensorizers_from_scratch,
        )
        self.filter_target_ood_slots = filter_target_ood_slots
        self.merge_source_vocab = merge_source_vocab
        self.decoupled_func_source = (
            get_blind_decoupled if decoupled_source else (lambda x: x)
        )
        self.decoupled_func_target = (
            get_blind_decoupled
            if decoupled_target
            else (lambda x, filter_ood_slots=None: x)
        )

        # Don't mess with the tensorizers if they're being loaded from a saved state.
        if init_tensorizers:
            # Unify special tokens across encoder and decoder.
            new_specials = [unk_token, pad_token, bos_token, eos_token, mask_token]
            src_vocab = self.tensorizers["src_seq_tokens"].vocab
            trg_vocab = self.tensorizers["trg_seq_tokens"].vocab
            self._replace_tokens(src_vocab, *new_specials)
            self._replace_tokens(trg_vocab, *new_specials)

            # Tensorizers inheriting from BertSeqLabelingTensorizerBase use a class-level
            # bos_token to encode inputs, so update the src tensorizer's bos_token if it exists.
            if hasattr(self.tensorizers["src_seq_tokens"], "bos_token"):
                self.tensorizers["src_seq_tokens"].bos_token = bos_token

            # Merge source and target vocabs, keeping them aligned. This is required
            # by the implementation of the pointer mechanism in the model's decoder.
            src_vocab = self.tensorizers["src_seq_tokens"].vocab
            trg_vocab = self.tensorizers["trg_seq_tokens"].vocab
            tokens_not_in_src = set(trg_vocab._vocab).difference(set(src_vocab._vocab))
            merged_tokens = src_vocab._vocab.copy() + [
                w for w in trg_vocab._vocab if w in tokens_not_in_src
            ]  # Order stays consistent with trg vocab. No randomness from set.
            merged_vocab = Vocabulary(
                vocab_list=merged_tokens,
                replacements=None,
                unk_token=unk_token,
                pad_token=pad_token,
                bos_token=bos_token,
                eos_token=eos_token,
                mask_token=mask_token,
            )
            print(f"Source vocab: {len(src_vocab)} entries.")
            print(f"Target vocab: {len(trg_vocab)} entries.")
            print(f"Merged vocab: {len(merged_vocab)} entries.")

            if self.merge_source_vocab:
                self.tensorizers["src_seq_tokens"].vocab = merged_vocab
                print("\tInitialized source tensorizer with merged vocab.")

            self.tensorizers["trg_seq_tokens"].vocab = merged_vocab
            print("\tInitialized target tensorizer with merged vocab.")

    def numberize_rows(self, rows):
        source_column = getattr(self.tensorizers["src_seq_tokens"], "text_column", None)
        if not source_column:
            source_column = self.tensorizers["src_seq_tokens"].columns[0]
        target_column = getattr(self.tensorizers["trg_seq_tokens"], "text_column", None)
        if not target_column:
            target_column = self.tensorizers["trg_seq_tokens"].columns[0]
        for idx, row in enumerate(rows):
            try:
                # Create a version of `row` where the source/target columns are
                # potentially decoupled. Everything else is copied over.
                res = row.copy()
                res[source_column] = self.decoupled_func_source(row[source_column])
                res[target_column] = self.decoupled_func_target(
                    row[target_column], filter_ood_slots=self.filter_target_ood_slots
                )

                numberized = {
                    name: tensorizer.numberize(res)
                    for name, tensorizer in self.tensorizers.items()
                }

                yield RowData(res, numberized)
            except ValueError:
                print(f"Skipping row #{idx}: {row}")
                continue
