#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import random

import click


VALIDATION_SPLIT = 0.25
SAMPLE_PRINT_COUNT = 5


def read_vocab(file_path):
    """
    Given a file, prepare the vocab dictionary where each line is the value and
    (line_no - 1) is the key
    """
    vocab = {}
    with open(file_path, "r") as file_contents:
        for idx, word in enumerate(file_contents):
            vocab[idx] = word.strip()
    return vocab


def stringify(sentence, vocab):
    """
    Given a numericalized sentence, fetch the correct word from the vocab and
    return it along with token indices in the new sentence

    Example:
    sentence :
        "1 2 3"
    vocab :
        {
            1 : "Get"
            2 : "me"
            3 : "water"
        }

    return value :
        [
            ["Get", [0, 3]],
            ["me", [4, 6]],
            ["water", [7, 12]],
        ]
    """
    return_val = []
    length_so_far = -1
    for idx in sentence.strip().split(" "):
        beg_index = length_so_far + 1
        word = vocab[int(idx)]
        end_index = beg_index + len(word)
        length_so_far = end_index
        return_val.append([word, [beg_index, end_index]])
    return return_val


def extract_slot_name(slot):
    """
    Two valid slots under IOB scheme are B-Slot_Name / I-Slot_Name
    This function returns the Slot_Name
    """
    if is_valid_slot(slot):
        return slot[2:]
    else:
        raise Exception("Invalid slot")


def is_valid_slot(slot):
    return slot.startswith("B-") or slot.startswith("I-")


def get_all_slots(slot_vals, query_vals):
    all_slots = []
    open_slot = []
    for _idx, (slot, query) in enumerate(zip(slot_vals, query_vals)):
        if open_slot:
            if not is_valid_slot(slot[0]):
                slot_beg_index = open_slot[0][1][0]
                slot_end_index = open_slot[-1][1][1]
                all_slots.append(
                    str(slot_beg_index)
                    + ":"
                    + str(slot_end_index)
                    + ":"
                    + extract_slot_name(open_slot[0][0])
                )
                open_slot.clear()
                continue
        if is_valid_slot(slot[0]):
            open_slot.append([slot[0], query[1]])
    return ",".join(all_slots)


def process_train_set(
    download_folder, output_directory, intent_vocab, slots_vocab, words_vocab
):
    train_file_name = os.path.join(output_directory, "atis.processed.train.csv")
    validation_file_name = os.path.join(output_directory, "atis.processed.val.csv")

    with open(
        os.path.join(download_folder, "atis.train.intent.csv"), "r"
    ) as intents, open(
        os.path.join(download_folder, "atis.train.slots.csv"), "r"
    ) as slots, open(
        os.path.join(download_folder, "atis.train.query.csv"), "r"
    ) as queries, open(
        train_file_name, "w"
    ) as train_file, open(
        validation_file_name, "w"
    ) as validation_file:

        for _idx, (intent, slot, query) in enumerate(zip(intents, slots, queries)):
            # There is always one intent for an utterance
            intent_string = stringify(intent, intent_vocab)[0][0]
            slot_vals = stringify(slot, slots_vocab)
            query_vals = stringify(query, words_vocab)

            slots_string = get_all_slots(slot_vals, query_vals)
            raw_utterance = " ".join([word[0] for word in query_vals])
            if random.random() < VALIDATION_SPLIT:
                validation_file.write(
                    intent_string + "\t" + slots_string + "\t" + raw_utterance + "\n"
                )
            else:
                train_file.write(
                    intent_string + "\t" + slots_string + "\t" + raw_utterance + "\n"
                )
    return [train_file_name, validation_file_name]


def process_test_set(
    download_folder, output_directory, intent_vocab, slots_vocab, words_vocab
):
    test_file_name = os.path.join(output_directory, "atis.processed.test.csv")

    with open(
        os.path.join(download_folder, "atis.test.intent.csv"), "r"
    ) as intents, open(
        os.path.join(download_folder, "atis.test.slots.csv"), "r"
    ) as slots, open(
        os.path.join(download_folder, "atis.test.query.csv"), "r"
    ) as queries, open(
        test_file_name, "w"
    ) as test_file:
        for _idx, (intent, slot, query) in enumerate(zip(intents, slots, queries)):
            # There is always one intent for an utterance
            intent_string = stringify(intent, intent_vocab)[0][0]
            slot_vals = stringify(slot, slots_vocab)
            query_vals = stringify(query, words_vocab)

            slots_string = get_all_slots(slot_vals, query_vals)
            raw_utterance = " ".join([word[0] for word in query_vals])
            test_file.write(
                intent_string + "\t" + slots_string + "\t" + raw_utterance + "\n"
            )
    return [test_file_name]


def print_sample(file_name):
    with open(file_name, "r") as given_file:
        for _i in range(SAMPLE_PRINT_COUNT):
            line = next(given_file).strip()
            print(line)


@click.command()
@click.option("-d", "--download-folder", required=True, type=str)
@click.option("-o", "--output-directory", required=True, type=str)
@click.option("-v", "--verbose", default=False, type=bool)
def main(download_folder, output_directory, verbose):
    intent_vocab = read_vocab(os.path.join(download_folder, "atis.dict.intent.csv"))
    slots_vocab = read_vocab(os.path.join(download_folder, "atis.dict.slots.csv"))
    words_vocab = read_vocab(os.path.join(download_folder, "atis.dict.vocab.csv"))

    train_val_files = process_train_set(
        download_folder, output_directory, intent_vocab, slots_vocab, words_vocab
    )
    if train_val_files:
        print(
            "Train/Validation data written successfully at ",
            " and ".join(train_val_files),
        )

    test_file = process_test_set(
        download_folder, output_directory, intent_vocab, slots_vocab, words_vocab
    )
    if test_file:
        print("Test data written successfully at ", test_file[0])

    if verbose is True:
        for file_name in train_val_files + test_file:
            print("\nSample row from file ", file_name)
            print_sample(file_name)


if __name__ == "__main__":
    main()
