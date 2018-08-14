#!/usr/bin/env python3
import datetime
from collections import OrderedDict
from random import shuffle
from typing import List

import libfb.py.fbpkg as fbpkg
import numpy as np
import pytext.utils.cuda_utils as cuda_utils
import read_data
import torch as torch
import torch.multiprocessing as mp
from agreement_calculator import (
    ConfusionMatrix,
    Label_and_Span_Calculator,
    Strict_Label_and_Span_Calculator,
)
from Parser import RNNGParser as parser_py
from pytext.args import parse_config
from pytext.config import PyTextConfig, config_to_json
from pytext.config.field_config import EmbedInitStrategy
from pytext.jobspec import SemanticParsingJobSpec
from pytext.optimizers import create_optimizer, optimizer_step, optimizer_zero_grad
from pytext.rnng.annotation import tree_from_tokens_and_indx_actions
from pytext.rnng.predict_parser import load_model
from pytext.rnng.rnng_cpp.caffe2_utils import get_model_config
from pytext.rnng.rnng_cpp.predict import RNNGPredictor_CPP
from pytext.utils import embeddings_utils
from rnng_bindings import Parser as parser_cpp
from utils import BiDict


STRICT_METRIC = Strict_Label_and_Span_Calculator()
PARTIAL_METRIC = Label_and_Span_Calculator(deletion=False)


def read_embeddings(word_bidict: BiDict, embeddings_path: str, embed_dim: int):
    print("reading embeddings from {}".format(embeddings_path))
    pretrained_embeds = embeddings_utils.PretrainedEmbedding(embeddings_path)
    embed_tokens_to_ids = {}
    for i in range(word_bidict.size()):
        word = word_bidict.value(i)
        if word in pretrained_embeds.embed_vocab:
            embed_tokens_to_ids[word] = i
        else:
            print(word + " is not in the word embedddings")
    token_embeddings = pretrained_embeds.initialize_embeddings_weights(
        embed_tokens_to_ids, -1, word_bidict.size(), embed_dim, EmbedInitStrategy.RANDOM
    )
    return token_embeddings


def train_rnng(config, learning_curve_reporter=None):
    cuda_utils.CUDA_ENABLED = config.use_cuda_if_available and torch.cuda.is_available()

    debug_filename = config.debug_path
    run(
        training_file=config.train_file_path,
        dev_file=config.eval_file_path,
        test_file=config.test_file_path,
        debug_file=debug_filename,
        embeddings_path=config.jobspec.data_handler.pretrained_embeds_file,
        model_snapshot_path=config.save_snapshot_path,
        config=config,
        learning_curve_reporter=learning_curve_reporter,
    )


def eval_rnng(config):
    cuda_utils.CUDA_ENABLED = config.use_cuda_if_available and torch.cuda.is_available()

    model_dir = config.save_snapshot_path

    rnng_config = config.jobspec.model
    if rnng_config.use_cpp:
        rnng_predictor = RNNGPredictor_CPP(model_dir)
    else:
        rnng_predictor = load_model(model_dir)

    if config.jobspec.features.dict_feat:
        print("Using dictionary features.")
    test_taggedsents = read_data.read_annotated_file(
        filename=config.test_file_path,
        actions_bidict=rnng_predictor.actions_bidict,
        max_to_read=rnng_config.max_test_num,
        add_dict_feat=config.jobspec.features.dict_feat,
    )
    for tg in test_taggedsents:
        read_data.set_tokens_indices(
            tg.sentence, rnng_predictor.terminal_bidict, rnng_predictor.dictfeat_bidict
        )

    return eval_model(
        rnng_predictor.model,
        test_taggedsents=test_taggedsents,
        terminal_bidict=rnng_predictor.terminal_bidict,
        actions_bidict=rnng_predictor.actions_bidict,
        test_out_path=config.test_out_path,
        all_metrics=True,
    )


def run(
    training_file,
    dev_file,
    test_file,
    debug_file,
    embeddings_path,
    model_snapshot_path,
    config,
    learning_curve_reporter=None,
):
    rnng_config = config.jobspec.model
    if config.jobspec.features.dict_feat:
        print("Using dictionary features.")
    with open(debug_file, "w") as debug_w:
        (
            oracle_dicts,
            train_taggedsents,
            dev_taggedsents,
            test_taggedsents,
        ) = read_data.read_train_dev_test(
            train_filename=training_file,
            dev_filename=dev_file,
            test_filename=test_file,
            max_train_num=rnng_config.max_train_num,
            max_dev_num=rnng_config.max_dev_num,
            max_test_num=rnng_config.max_test_num,
            brackets="[]",
            add_dict_feat=config.jobspec.features.dict_feat,
        )

        print(
            "Done reading data; # train instances: {}, # dev instances: {},\
             # test instances: {}".format(
                len(train_taggedsents), len(dev_taggedsents), len(test_taggedsents)
            )
        )
        print(
            "# terminals: {}, # actions: {}, # dictfeat: {}".format(
                oracle_dicts.terminal_bidict.size(),
                oracle_dicts.actions_bidict.size(),
                oracle_dicts.dictfeat_bidict.size(),
            )
        )
        print("actions are " + str(oracle_dicts.actions_bidict.vocab()))
        print("dict feats are " + str(oracle_dicts.dictfeat_bidict.vocab()))

        if rnng_config.use_cpp:
            print("Training with C++")
            model = parser_cpp(
                get_model_config(config),
                oracle_dicts.actions_bidict.get_sorted_objs(),
                oracle_dicts.terminal_bidict.get_sorted_objs(),
                oracle_dicts.dictfeat_bidict.get_sorted_objs()
                if oracle_dicts.dictfeat_bidict
                else [],  # [] because we can't pass NULL to C++ Parser.
            ).make()
        else:
            print("Training with Python")
            model = parser_py(
                config=config,
                terminal_bidict=oracle_dicts.terminal_bidict,
                actions_bidict=oracle_dicts.actions_bidict,
                dictfeat_bidict=oracle_dicts.dictfeat_bidict,
            )

        pretrained_word_weights = None
        if embeddings_path:
            pretrained_word_weights = read_embeddings(
                word_bidict=oracle_dicts.terminal_bidict,
                embeddings_path=embeddings_path,
                embed_dim=config.jobspec.features.word_feat.embed_dim,
            )
            model.init_word_weights(pretrained_word_weights)

        print_debug(
            "Done making model. Now training the model with learning rate={}".format(
                config.jobspec.optimizer.lr
            ),
            debug_w,
        )

        if cuda_utils.CUDA_ENABLED:
            print("Training on GPU")
            model.cuda()
        else:
            print("Training on CPU")

        print_debug(
            "Num workers is {}".format(config.jobspec.trainer.num_workers), debug_w
        )

        # cpp case
        if rnng_config.use_cpp:
            train_eval_epochs(
                model,
                train_taggedsents,
                dev_taggedsents,
                oracle_dicts,
                model_snapshot_path,
                debug_w,
                config,
                0,
                rnng_config,
                learning_curve_reporter,
            )
        # keep parent process for update, spawn n-1 child processes
        else:
            model.share_memory()
            processes = []
            for n_worker in range(1, config.jobspec.trainer.num_workers):
                p = mp.Process(
                    target=train_eval_epochs,
                    args=(
                        model,
                        train_taggedsents,
                        dev_taggedsents,
                        oracle_dicts,
                        model_snapshot_path,
                        debug_w if n_worker == 0 else None,
                        config,
                        n_worker,
                        rnng_config,
                    ),
                )
                p.start()
                processes.append(p)

            train_eval_epochs(
                model,
                train_taggedsents,
                dev_taggedsents,
                oracle_dicts,
                model_snapshot_path,
                debug_w,
                config,
                0,
                rnng_config,
                learning_curve_reporter,
            )

            for p in processes:
                p.join()

        print("Done training the model.")


def train_eval_epochs(  # noqa: C901
    model,
    train_taggedsents,
    dev_taggedsents,
    oracle_dicts,
    model_snapshot_path,
    debug_w,
    config,
    rank,
    rnng_config,
    learning_curve_reporter=None,
):
    torch.set_num_threads(1)
    torch.manual_seed(config.jobspec.trainer.random_seed + rank)
    max_num_epochs = config.jobspec.trainer.epochs

    optimizers = create_optimizer(model, config.jobspec.optimizer)

    epochs = np.zeros(max_num_epochs)
    losses = np.zeros(max_num_epochs)

    dev_epochs = []
    dev_losses = []

    dev_frame_accuracy = []
    total_instances = len(train_taggedsents)

    best_frame_acc = -1

    for num_epoch in range(max_num_epochs):
        print("\nNum epoch {}/{}".format(num_epoch + 1, max_num_epochs))
        if config.jobspec.data_handler.shuffle:
            shuffle(train_taggedsents)

        model.train()
        total_loss = 0.
        num_instances = 0
        start_time = datetime.datetime.now().replace(microsecond=0)

        for tagged_sent in train_taggedsents:

            num_instances += 1
            actions_idx_rev = tagged_sent.actions_idx_rev
            tokens_indices_rev = tagged_sent.sentence.indices_rev
            dictfeat_indices_rev = tagged_sent.sentence.dictfeat_indices_rev
            dictfeat_weights = tagged_sent.sentence.dict_feat_weights
            dictfeat_lengths = tagged_sent.sentence.dict_feat_lengths
            try:
                result = model.forward(
                    [
                        cuda_utils.Variable(torch.LongTensor(tokens_indices_rev)),
                        cuda_utils.Variable(torch.LongTensor(dictfeat_indices_rev)),
                        cuda_utils.Variable(torch.FloatTensor(dictfeat_weights[::-1])),
                        cuda_utils.Variable(torch.LongTensor(dictfeat_lengths[::-1])),
                        cuda_utils.Variable(torch.LongTensor(actions_idx_rev)),
                    ]
                )
            except Exception as err:
                print("Tagged sent is " + str(tagged_sent.sentence.raw))
                print("actions indices are " + str(actions_idx_rev))
                raise err

            action_scores = result[1]
            loss = compute_loss(action_scores, actions_idx_rev[::-1])
            total_loss += float(loss)
            optimizer_zero_grad(optimizers)
            loss.backward()
            # TODO: look into this
            # model.reset_hidden()
            optimizer_step(optimizers)
            if rank == 0 and num_instances % 1000 == 0:
                delta_time = datetime.datetime.now().replace(microsecond=0) - start_time
                print(
                    "[%d mins] Processed [%d/%d]\r"
                    % (delta_time.total_seconds() / 60, num_instances, total_instances),
                    end="",
                    flush=True,
                )

        epochs[num_epoch] = num_epoch
        avg_loss = total_loss / num_instances
        losses[num_epoch] = avg_loss

        time_spent = datetime.datetime.now().replace(microsecond=0) - start_time
        print_debug(
            "train epoch %d: avg loss: %.6f, time: %s secs, avg time: %s secs"
            % (
                num_epoch + 1,
                avg_loss,
                str(time_spent),
                str(time_spent / num_instances),
            ),
            debug_w,
        )

        # evaluate only for one worker
        if rank == 0 and (
            num_epoch % config.jobspec.trainer.eval_interval == 0
            or num_epoch == max_num_epochs - 1
        ):
            evaluation, eval_avg_loss, frame_accuracy = eval_model(
                model=model,
                test_taggedsents=dev_taggedsents,
                terminal_bidict=oracle_dicts.terminal_bidict,
                actions_bidict=oracle_dicts.actions_bidict,
                test_out_path=config.test_out_path + "_" + str(num_epoch),
                num_epoch=num_epoch,
                print_incorrect=False,
                all_metrics=rnng_config.all_metrics,
                debug_w=debug_w,
            )
            if frame_accuracy > best_frame_acc:
                best_frame_acc = frame_accuracy
                if model_snapshot_path is not None:

                    snapshot = OrderedDict(
                        [
                            ("model_state", model.state_dict()),
                            ("pytext_config", config_to_json(PyTextConfig, config)),
                            ("oracle_dicts", oracle_dicts),
                            ("compositional", True),
                        ]
                    )  # type: OrderedDict
                    torch.save(snapshot, model_snapshot_path)

                    print("Saved model at {}".format(model_snapshot_path))

            if rnng_config.all_metrics:
                evaluation.print_eval()

            print("Best frame accuracy yet: {}".format(best_frame_acc))
            dev_losses.append(eval_avg_loss)
            dev_frame_accuracy.append(frame_accuracy)
            dev_epochs.append(num_epoch)

            # Update the learning curve now
            if learning_curve_reporter is not None:
                learning_curve_reporter(num_epoch, eval_avg_loss, frame_accuracy)

        if debug_w is not None:
            debug_w.flush()


def eval_model(
    model,
    test_taggedsents: List[read_data.TaggedSentence],
    terminal_bidict: BiDict,
    actions_bidict: BiDict,
    test_out_path: str = None,
    num_epoch: int = 0,
    print_incorrect=False,
    all_metrics=False,
    debug_w=None,
):
    if cuda_utils.CUDA_ENABLED:
        model.cuda()
    model.eval()

    dev_instances = 0
    dev_loss = 0.
    num_correct_topintent = 0.
    correct_frame = 0.

    confusion_matrix = ConfusionMatrix()
    partial_confusion_matrix = ConfusionMatrix()
    test_out_w = open(test_out_path, "w") if test_out_path is not None else None

    for tagged_sent_test in test_taggedsents:
        dev_instances += 1
        tokens_indices_rev = tagged_sent_test.sentence.indices_rev
        actions_idx_rev_dev = tagged_sent_test.actions_idx_rev
        dictfeat_indices_rev = tagged_sent_test.sentence.dictfeat_indices_rev
        dictfeat_weights = tagged_sent_test.sentence.dict_feat_weights
        dictfeat_lengths = tagged_sent_test.sentence.dict_feat_lengths

        result = model.forward(
            [
                cuda_utils.Variable(torch.LongTensor(tokens_indices_rev)),
                cuda_utils.Variable(torch.LongTensor(dictfeat_indices_rev)),
                cuda_utils.Variable(torch.FloatTensor(dictfeat_weights[::-1])),
                cuda_utils.Variable(torch.LongTensor(dictfeat_lengths[::-1])),
                cuda_utils.Variable(torch.LongTensor(actions_idx_rev_dev)),
            ]
        )
        actions_taken_idx = result[0].data.cpu().numpy()
        action_scores = result[1]

        loss = compute_loss(action_scores, actions_idx_rev_dev[::-1])

        pred_tree = None
        if test_out_w is not None:
            pred_tree = tree_from_tokens_and_indx_actions(
                tagged_sent_test.sentence.raw, actions_bidict, actions_taken_idx
            )
            test_out_w.write(pred_tree.flat_str() + "\n")

        if all_metrics:
            gold_tree = tree_from_tokens_and_indx_actions(
                tagged_sent_test.sentence.raw, actions_bidict, actions_idx_rev_dev[::-1]
            )
            if pred_tree is None:
                pred_tree = tree_from_tokens_and_indx_actions(
                    tagged_sent_test.sentence.raw, actions_bidict, actions_taken_idx
                )

            STRICT_METRIC.tree_similarity((gold_tree, pred_tree), confusion_matrix)

            PARTIAL_METRIC.tree_similarity(
                (gold_tree, pred_tree), partial_confusion_matrix
            )

        if actions_taken_idx[0] == actions_idx_rev_dev[-1]:
            num_correct_topintent += 1
        if np.array_equal(actions_taken_idx, actions_idx_rev_dev[::-1]):
            correct_frame += 1

        if print_incorrect and not actions_taken_idx == actions_idx_rev_dev[::-1]:
            print("\nTokens: " + str(tagged_sent_test.sentence.raw))
            print("Actions: " + str(actions_taken_idx))

        if loss is not None:
            dev_loss += float(loss)

    if all_metrics:
        strict_evaluation = confusion_matrix.compute_metrics()
        partial_evaluation = partial_confusion_matrix.compute_metrics()
        all_metrics_str = "strict-f1 score: %.2f, partial-f1 score: %.2f" % (
            strict_evaluation.all_scores.f1 * 100,
            partial_evaluation.all_scores.f1 * 100,
        )
    else:
        strict_evaluation = confusion_matrix.compute_metrics()
        all_metrics_str = ""

    frame_accuracy = correct_frame / dev_instances * 100
    avg_loss = dev_loss / dev_instances  # noqa: T484
    print_debug(
        "[valid/test] epoch %d: per-instance loss: %.6f, top-intent accuracy: %.4f, \
        frame accuracy: %.2f, %s"
        % (
            num_epoch + 1,
            avg_loss,
            num_correct_topintent / dev_instances * 100,
            frame_accuracy,
            all_metrics_str,
        ),
        debug_w,
    )

    if test_out_w is not None:
        test_out_w.close()

    return strict_evaluation, avg_loss, frame_accuracy


def compute_loss(action_scores, targets):
    # action scores is a 2D Tensor of dims sequence_length x number_of_actions
    # targets is a 1D list of integers of length sequence_length

    criterion = torch.nn.CrossEntropyLoss()
    action_scores_list = torch.chunk(action_scores, action_scores.size()[0])
    target_vars = [cuda_utils.Variable(torch.LongTensor([t])) for t in targets]
    losses = [
        criterion(action, target).view(1)
        for action, target in zip(action_scores_list, target_vars)
    ]
    total_loss = torch.sum(torch.cat(losses)) if len(losses) > 0 else None
    return total_loss


def print_debug(l, debug_w=None):
    print(l)
    if debug_w is not None:
        debug_w.write(str(l) + "\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    config = parse_config()
    if config.jobspec.data_handler.pretrained_embeds_file:
        print("Fetching embedding pkg")
        pretrained_embeds_file = fbpkg.fetch(
            config.jobspec.data_handler.pretrained_embeds_file,
            dst="/tmp",
            verbose=False,
        )
        config.jobspec.data_handler.pretrained_embeds_file = pretrained_embeds_file
    assert isinstance(config.jobspec, SemanticParsingJobSpec)
    train_rnng(config)
