#!/usr/bin/env python3
from typing import List, Dict, Set
import json
from pytext.rnng.utils import (
    BiDict,
    is_intent_nonterminal,
    is_slot_nonterminal,
)


class Ontology:

    def __init__(self, filename: str) -> None:
        """
        Read and store an ontology from a JSON file.

        The JSON format is as in the example below:
        {
            "domains": {
                "navigation": {
                    "intents": {
                        "IN:GET_DIRECTION": { "valid_slots": ["SL:DESTINATION", ...] },
                        "IN:GET_LOCATION": { "valid_slots": ["SL:ATTRIBUTE", ...] },
                        ...
                    }
                },
                ...
            }
        }
        """
        print("Reading ontology from JSON: {}".format(filename))
        with open(filename) as fin:
            raw = json.load(fin)
        self.intent_to_valid_slots: Dict[str, Set[str]] = {}
        for domain_schema in raw["domains"].values():
            for intent_name, intent_schema in domain_schema["intents"].items():
                valid_slots = set(intent_schema["valid_slots"])
                self.intent_to_valid_slots[intent_name] = valid_slots


class OntologyConstraint:

    def __init__(
        self,
        ontology_filename: str,
        actions_bidict: BiDict,
    ) -> None:
        """
        Read an ontology and provide constraints on the actions during parsing.
        """
        self.ontology = Ontology(ontology_filename)
        self.actions_bidict = actions_bidict
        self.intent_idx_to_slot_idxs: Dict[int, List[int]] = {}
        self._preprocess()

    def _preprocess(self):
        all_slot_idxs = [
            self.actions_bidict.index(nt)
            for nt in self.actions_bidict.vocab()
            if is_slot_nonterminal(nt)
        ]
        for nt in self.actions_bidict.vocab():
            if not is_intent_nonterminal(nt):
                continue
            nt_idx = self.actions_bidict.index(nt)
            if nt not in self.ontology.intent_to_valid_slots:
                print("Intent {} not found in ontology.".format(nt))
                self.intent_idx_to_slot_idxs[nt_idx] = all_slot_idxs
            else:
                valid_slots = [
                    sl for sl in self.ontology.intent_to_valid_slots[nt]
                    if self.actions_bidict.check(sl)
                ]
                print("Intent {} -> Slots {}".format(nt, valid_slots))
                self.intent_idx_to_slot_idxs[nt_idx] = [
                    self.actions_bidict.index(sl) for sl in valid_slots
                ]

    def valid_SL_for_IN(self, IN_idx) -> List[int]:
        """
        Return the indices of slot actions that are valid under the given intent.
        If the intent is not in the ontology, return all slot actions.
        """
        return self.intent_idx_to_slot_idxs[int(IN_idx)]
