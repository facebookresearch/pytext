#!/usr/bin/env python3
import argparse
from typing import List
from collections import Counter
from pytext.rnng.utils import BiDict, UNKNOWN_WORD, NUM, is_number
from pytext.rnng.annotation import Annotation

EMPTY_BIDICT = BiDict()


class Sentence:
    def __init__(
        self,
        raw,
        lowercase,
        dict_feats,
        dict_feat_weights,
        dict_feat_lengths,
    ):
        self.raw = raw
        self.lowercase = lowercase
        self.dict_feats = dict_feats
        self.dict_feat_weights = dict_feat_weights
        self.dict_feat_lengths = dict_feat_lengths
        # UNKified
        self.input_tokens = None
        self.indices_rev = None
        self.dictfeat_indices_rev = None

    def set_input_tokens(self, input_tokens, indices_rev, dictfeat_indices_rev=None):
        assert len(input_tokens) == len(
            self.raw
        ), "current code base, esp tree output part assumes same length"

        self.input_tokens = input_tokens
        self.indices_rev = indices_rev
        self.dictfeat_indices_rev = dictfeat_indices_rev

    def size(self):
        return len(self.raw)

    def __str__(self):
        return str(self.raw)


class TaggedSentence:
    def __init__(
        self, sentence, actions_rev, actions_idx_rev, utterance, doc_index
    ):
        self.sentence = sentence
        self.actions_rev = actions_rev
        self.actions_idx_rev = actions_idx_rev
        self.utterance = utterance
        self.doc_index = doc_index

    def __str__(self):
        return "( {}; {} )".format(str(self.sentence), str(self.actions_rev[::-1]))

    def __repr__(self):
        return self.__str__()


class Oracle_Dicts:
    def __init__(self):
        self.terminal_bidict = BiDict()
        self.actions_bidict = BiDict()
        self.nonterminal_bidict = BiDict()
        self.dictfeat_bidict = BiDict()

    def add_sent(self, tagged_sent):
        self.tagged_sentences.append(tagged_sent)

    def __str__(self):
        return str(self.tagged_sentences)


def read_train_dev_test(
    train_filename,
    dev_filename,
    test_filename,
    brackets="[]",
    max_train_num=None,
    max_dev_num=None,
    max_test_num=None,
    add_dict_feat=False,
):
    oracle_dicts = Oracle_Dicts()
    train_freq_counter = Counter()
    dict_feat_counter = Counter()
    if train_filename is not None:
        train_taggedsents = read_annotated_file(
            filename=train_filename,
            actions_bidict=oracle_dicts.actions_bidict,
            max_to_read=max_train_num,
            brackets=brackets,
            freq_counter=train_freq_counter,
            dict_feat_counter=dict_feat_counter,
            add_dict_feat=add_dict_feat,
        )
    else:
        raise TypeError("Need to provide training file")

    oracle_dicts.terminal_bidict = get_dict_from_freq(train_freq_counter)
    if add_dict_feat:
        print("Reading dict feat from data.")
        oracle_dicts.dictfeat_bidict = get_dict_from_freq(
            dict_feat_counter, is_dict=True, min_freq=0
        )

    for tg in train_taggedsents:
        set_tokens_indices(
            tg.sentence, oracle_dicts.terminal_bidict, oracle_dicts.dictfeat_bidict
        )

    dev_taggedsents = (
        read_annotated_file(
            filename=dev_filename,
            actions_bidict=oracle_dicts.actions_bidict,
            max_to_read=max_dev_num,
            brackets=brackets,
            add_dict_feat=add_dict_feat,
        )
        if dev_filename is not None
        else []
    )

    for tg in dev_taggedsents:
        set_tokens_indices(
            tg.sentence, oracle_dicts.terminal_bidict, oracle_dicts.dictfeat_bidict
        )

    test_taggedsents = (
        read_annotated_file(
            filename=test_filename,
            actions_bidict=oracle_dicts.actions_bidict,
            max_to_read=max_test_num,
            brackets=brackets,
            add_dict_feat=add_dict_feat,
        )
        if test_filename is not None
        else []
    )

    for tg in test_taggedsents:
        set_tokens_indices(
            tg.sentence, oracle_dicts.terminal_bidict, oracle_dicts.dictfeat_bidict
        )

    return (oracle_dicts, train_taggedsents, dev_taggedsents, test_taggedsents)


def get_dict_from_freq(train_freq_counter, is_dict=False, min_freq=1):
    bidict = BiDict()
    for word in train_freq_counter:
        if train_freq_counter[word] > min_freq:
            bidict.check_add(word)
        elif not is_dict:
            unk_token = unkify(word, bidict)
            bidict.check_add(unk_token)
    bidict.check_add(UNKNOWN_WORD)
    return bidict


# this function is copied from
# https://github.com/clab/rnng/blob/master/get_oracle.py
def unkify(token, words_bidict):  # noqa: C901
    if len(token.rstrip()) == 0:
        return UNKNOWN_WORD

    if words_bidict.check(token):
        return token

    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False
    for char in token.rstrip():
        if char.isdigit():
            hasDigit = True
        elif char == "-":
            hasDash = True
        elif char.isalpha():
            if char.islower():
                hasLower = True
            elif char.isupper():
                numCaps += 1
    result = UNKNOWN_WORD
    lower = token.rstrip().lower()
    ch0 = token.rstrip()[0]
    if ch0.isupper():
        if numCaps == 1:
            result = result + "-INITC"
            if words_bidict.check(lower):
                result = result + "-KNOWNLC"
        else:
            result = result + "-CAPS"
    elif not (ch0.isalpha()) and numCaps > 0:
        result = result + "-CAPS"
    elif hasLower:
        result = result + "-LC"
    if hasDigit:
        result = result + "-NUM"
    if hasDash:
        result = result + "-DASH"
    if lower[-1] == "s" and len(lower) >= 3:
        ch2 = lower[-2]
        if not (ch2 == "s") and not (ch2 == "i") and not (ch2 == "u"):
            result = result + "-s"
    elif len(lower) >= 5 and not (hasDash) and not (hasDigit and numCaps > 0):
        if lower[-2:] == "ed":
            result = result + "-ed"
        elif lower[-3:] == "ing":
            result = result + "-ing"
        elif lower[-3:] == "ion":
            result = result + "-ion"
        elif lower[-2:] == "er":
            result = result + "-er"
        elif lower[-3:] == "est":
            result = result + "-est"
        elif lower[-2:] == "ly":
            result = result + "-ly"
        elif lower[-3:] == "ity":
            result = result + "-ity"
        elif lower[-1] == "y":
            result = result + "-y"
        elif lower[-2:] == "al":
            result = result + "-al"
    return result


def indices(toks: List[str], bi_dict: BiDict) -> List[int]:
    return (
        [
            bi_dict.index(x) if bi_dict.check(x) else bi_dict.index(UNKNOWN_WORD)
            for x in toks
        ]
        if all([toks, bi_dict, bi_dict.size()])
        else []
    )


def set_tokens_indices(
    sent: Sentence, terminal_dict: BiDict, dictfeat_dict: BiDict = EMPTY_BIDICT
) -> None:
    in_toks = [unkify(x, terminal_dict) for x in sent.lowercase]
    sent.set_input_tokens(
        in_toks,
        indices(in_toks, terminal_dict)[::-1],
        indices(sent.dict_feats, dictfeat_dict)[::-1],
    )


def read_annotated_file(
    filename: str,
    actions_bidict: BiDict,
    max_to_read: int = None,
    brackets: str = "[]",
    freq_counter: Counter = None,
    dict_feat_counter: Counter = None,
    add_dict_feat=False,
) -> List[TaggedSentence]:
    # remember to call set_tokens_indices after this
    num_sent = 0
    doc_index = -1
    tagged_sentences: list = []
    with open(filename) as f:
        for line in f:
            if max_to_read and max_to_read > 0 and num_sent >= max_to_read:
                break
            line = line.strip()
            if line.startswith("#"):
                continue

            if len(line) == 0:
                break

            doc_index += 1

            try:
                annotation_tree = Annotation(
                    line, brackets=brackets, add_dict_feat=add_dict_feat
                )
            except ValueError as e:
                print(e)
                print("Ignoring invalid line: " + line)
                continue

            terminals = annotation_tree.tree.list_tokens()
            actions = annotation_tree.tree.to_actions()
            utterance = annotation_tree.utterance
            if add_dict_feat and annotation_tree.model_feats is not None:
                dict_feats = annotation_tree.model_feats.dictFeats
                dict_feat_weights = annotation_tree.model_feats.dictFeatWeights
                dict_feat_lengths = annotation_tree.model_feats.dictFeatLengths
            else:
                dict_feats, dict_feat_weights, dict_feat_lengths = [], [], []
            for action in actions:
                actions_bidict.check_add(action)

            lc_withnum = [NUM if is_number(x) else x.lower() for x in terminals]
            sentence = Sentence(
                terminals,
                lc_withnum,
                dict_feats,
                dict_feat_weights,
                dict_feat_lengths,
            )

            for x in lc_withnum:
                if freq_counter is not None:
                    freq_counter[x] += 1

            if add_dict_feat:
                for d in dict_feats:
                    if dict_feat_counter is not None:
                        dict_feat_counter[d] += 1

            actions.reverse()
            actions_idx_rev = [actions_bidict.index(a) for a in actions]
            tagged_sentence = TaggedSentence(
                sentence=sentence,
                actions_rev=actions,
                actions_idx_rev=actions_idx_rev,
                utterance=utterance,
                doc_index=doc_index,
            )
            tagged_sentences.append(tagged_sentence)
            num_sent += 1
    print("Read %d examples from %s" % (len(tagged_sentences), filename))
    return tagged_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read dataset")
    parser.add_argument("-filename", type=str, help="oracle file")
    args = parser.parse_args()

    filename = args.filename
    oracle_dicts = Oracle_Dicts()
    read_annotated_file(filename, oracle_dicts.actions_bidict)
