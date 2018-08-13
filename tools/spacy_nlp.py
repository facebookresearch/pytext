#!/usr/bin/env python3
import argparse
import en_core_web_md
from typing import List, Tuple, Any, NamedTuple, Dict
from spacy.tokens import Doc


class SpacyTokenizerWrapper(object):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, text):
        words = []
        spaces = []
        tokenized_text = self.tokenizer.tokenize_with_ranges(text)
        for (t, (_, e)) in tokenized_text:
            words.append(t)
            if e == len(text) or text[e] == " ":
                spaces.append(True)
            else:
                spaces.append(False)

        return Doc(self.vocab, words=words, spaces=spaces)


class SpanLabel(NamedTuple):
    label: str
    start: int
    end: int


class DependencyLabel(NamedTuple):
    dep: str
    head_idx: int


class SpacyResult(NamedTuple):
    tokens_with_ranges: List[Tuple[Any, Tuple[Any, Any]]]
    named_entities: List[SpanLabel]
    pos_tags: List[SpanLabel]
    dependency_parse: Dict[int, DependencyLabel]


class SpacyNLP:
    # A Wrapper for Spacy NLP model descriped here: https://fburl.com/nnrsv08p
    def __init__(self, tokenizer=None, n_threads=1):
        self.nlp = en_core_web_md.load()
        self.n_threads = n_threads
        if tokenizer is not None:
            self.nlp.tokenizer = SpacyTokenizerWrapper(self.nlp.vocab, tokenizer)

    def annotate(self, texts: List[str]) -> List[SpacyResult]:
        results = []
        for result in self.nlp.pipe(texts, n_threads=self.n_threads):
            results.append(self.parse_result(result))
        return results

    def parse_result(self, result: Doc) -> SpacyResult:
        tokens_dict = {token.idx: i for i, token in enumerate(result)}

        return SpacyResult(
            tokens_with_ranges=[
                (token.text, (token.idx, token.idx + len(token))) for token in result
            ],
            named_entities=[
                SpanLabel(ent.label_, ent.start_char, ent.end_char)
                for ent in result.ents
            ],
            pos_tags=[
                SpanLabel(token.pos_, token.idx, token.idx + len(token))
                for token in result
            ],
            dependency_parse={
                i: DependencyLabel(token.dep_, tokens_dict[token.head.idx])
                for i, token in enumerate(result)
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spacy NLP Toolkit")
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    spacy_nlp = SpacyNLP()
    print("Spacy result: \n")
    print(spacy_nlp.annotate([args.text]))
