#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pytext.common.utils import eprint

from .pytext_config import LATEST_VERSION


ADAPTERS = {}
NOT_THERE = (None, None, None)


def register_adapter(from_version):
    def decorator(fn):
        if from_version in ADAPTERS:
            raise Exception(
                "Duplicated adapter from_version={}: '{}' and '{}'".format(
                    from_version, fn.__name__, ADAPTERS[from_version].__name__
                )
            )
        else:
            ADAPTERS[from_version] = fn
        return fn

    return decorator


def find_dicts_containing_key(json_config, key):
    if key in json_config:
        yield json_config
    for _, v in json_config.items():
        if hasattr(v, "__contains__") and hasattr(v, "items"):
            yield from find_dicts_containing_key(v, key)


def rename(json_config, old_name, new_name):
    for section in find_dicts_containing_key(json_config, old_name):
        value = section.pop(old_name)
        if new_name:
            section[new_name] = value


def is_type_specifier(json_dict):
    """If a config object is a class, it might have a level which is a type specifier,
    with one key corresponding to the name of whichever type it is. These types should
    not be explicitly named in the path."""
    # heuristic: one key, starting with uppercase character
    if len(json_dict) != 1:
        return False
    key = next(iter(json_dict))
    return key[0] == key[0].upper()


def find_parameter(config, path_str):
    # Recursively find path elements, skipping into type specifiers.
    # Return the value and its container so the value can be deleted.

    path = path_str.split(".")
    value = config
    container = None
    for segment in path:
        while is_type_specifier(value):
            container, value = value, next(iter(value.values()))
        if segment not in value:
            return NOT_THERE
        container, value = value, value[segment]
    return path[-1], container, value


def _create_path(config, path):
    # Recursively find path elements, skipping into type specifiers.
    # If any container isn't there, create a new empty object for it.
    # This will only be created if the
    value = config
    for segment in path:
        while is_type_specifier(value):
            value = next(iter(value.values()))
        if segment not in value:
            value[segment] = {}
        value = value[segment]
    while is_type_specifier(value):
        value = next(iter(value.values()))
    return value


def create_parameter(config, path_str, value):
    *path, param = path_str.split(".")
    new_container = _create_path(config, path)
    new_container[param] = value


def delete_parameter(config, path_str):
    param_name, container, _ = find_parameter(config, path_str)
    if container:
        container.pop(param_name, None)


def rename_parameter(config, old_path, new_path, transform=lambda x: x):
    """A powerful tool for writing config adapters, this allows you to specify
    a JSON-style path for an old and new config parameter. For instance

    rename_parameter(config, "task.data.epoch_size", "task.trainer.batches_per_epoch")

    will look through the config for task.data.epoch_size, including moving through
    explicitly specified types. If it's specified, it will delete the value and
    set it in task.trainer.num_batches_per_epoch instead, creating trainer as an empty
    dictionary if necessary."""

    found = find_parameter(config, old_path)
    if found is not NOT_THERE:
        param_name, container, old_value = found
        # Delete old value
        container.pop(param_name)
        # Update new value
        create_parameter(config, new_path, transform(old_value))

    return config


@register_adapter(from_version=0)
def v0_to_v1(json_config):
    # migrate optimizer and random_seed params
    [task] = json_config["task"].values()
    if (
        "optimizer" not in task
        or "Adam" in task["optimizer"]
        or "SGD" in task["optimizer"]
        or "NAG" in task["optimizer"]
    ) and ("trainer" not in task or "random_seed" not in task["trainer"]):
        return json_config

    if "trainer" in task and "random_seed" in task["trainer"]:
        json_config["random_seed"] = task["trainer"]["random_seed"]
        del task["trainer"]["random_seed"]
    if "optimizer" in task and not any(
        opt in task["optimizer"] for opt in ["Adam", "SGD", "NAG"]
    ):
        op_type = task["optimizer"].get("type", "adam")
        if op_type == "adam":
            op_config = {"Adam": {}}
            for key in ["lr", "weight_decay"]:
                if key in task["optimizer"]:
                    op_config["Adam"][key] = task["optimizer"][key]
        elif op_type == "sgd":
            op_config = {"SGD": {}}
            for key in ["lr", "momentum"]:
                if key in task["optimizer"]:
                    op_config["SGD"][key] = task["optimizer"][key]
        elif op_type == "nag":
            op_config = {"NAG": {}}
            for key in ["lr", "weight_decay", "momentum"]:
                if key in task["optimizer"]:
                    op_config["NAG"][key] = task["optimizer"][key]
        else:
            raise ValueError("Migration not supported for your optimizer")
        task["optimizer"] = op_config
    return json_config


@register_adapter(from_version=1)
def v1_to_v2(json_config):
    # migrate optimizer params
    [task] = json_config["task"].values()
    if (
        "scheduler" not in task
        or task["scheduler"] is None
        or task["scheduler"].get("type") is None
    ):
        return json_config
    op_type = task["scheduler"].get("type")
    if op_type == "step_lr":
        op_config = {"StepLR": {}}
        for key in ["step_size", "gamma"]:
            if key in task["scheduler"]:
                op_config["StepLR"][key] = task["scheduler"][key]
        task["scheduler"] = op_config
    elif op_type == "lm_fine_tuning":
        op_config = {"LmFineTuning": {}}
        for key in [
            "cut_frac",
            "ratio",
            "non_pretrained_param_groups",
            "lm_lr_multiplier",
            "lm_use_per_layer_lr",
            "lm_gradual_unfreezing",
            "last_epoch",
        ]:
            if key in task["scheduler"]:
                op_config["LmFineTuning"][key] = task["scheduler"][key]
        task["scheduler"] = op_config
    elif op_type == "reduce_lr_on_plateau":
        op_config = {"ReduceLROnPlateau": {}}
        for key in [
            "lower_is_better",
            "factor",
            "patience",
            "min_lr",
            "threshold",
            "threshold_is_absolute",
            "cooldown",
        ]:
            if key in task["scheduler"]:
                op_config["ReduceLROnPlateau"][key] = task["scheduler"][key]
        task["scheduler"] = op_config
    elif op_type == "cosine_annealing_lr":
        op_config = {"CosineAnnealingLR": {}}
        for key in ["t_max", "eta_min"]:
            if key in task["scheduler"]:
                op_config["CosineAnnealingLR"][key] = task["scheduler"][key]
        task["scheduler"] = op_config
    elif op_type == "exponential_lr":
        op_config = {"ExponentialLR": {}}
        for key in ["gamma"]:
            if key in task["scheduler"]:
                op_config["ExponentialLR"][key] = task["scheduler"][key]
        task["scheduler"] = op_config
    elif op_type == "none":
        del task["scheduler"]
    else:
        raise ValueError("Migration for your scheduler %s not supported." % op_type)
    return json_config


@register_adapter(from_version=2)
def v2_to_v3(json_config):
    """Optimizer and Scheduler configs used to be part of the task config,
    they now live in the trainer's config.
    """
    [task] = json_config["task"].values()
    for section_str in ["optimizer", "scheduler"]:
        if section_str in task:
            if "trainer" not in task:
                task["trainer"] = {}
            trainer = task["trainer"]
            # a hack to support an older hack:
            # some tasks like ensemble have a 'real_trainer' section inside trainer
            # that has the actual trainer config
            if "real_trainer" in trainer:
                real_trainer = trainer["real_trainer"]
                real_trainer[section_str] = task[section_str]
            else:
                trainer[section_str] = task[section_str]
            # remove from task config
            task.pop(section_str)

    return json_config


@register_adapter(from_version=3)
def v3_to_v4(json_config):
    """Key for provding the path for contextual token embedding has changed from
    `pretrained_model_embedding` to `contextual_token_embedding. This affects the
    `features` section of the config.
    """
    [task] = json_config["task"].values()
    old_key = "pretrained_model_embedding"
    new_key = "contextual_token_embedding"
    for section_str in ["features", "labels"]:
        if section_str in task:
            section = task[section_str]
            if section and old_key in section:
                section[new_key] = section[old_key]
                section.pop(old_key)

    return json_config


def deprecate(json_config, t):
    for section in find_dicts_containing_key(json_config, t):
        section[t + "_Deprecated"] = section.pop(t)


@register_adapter(from_version=4)
def doc_model_deprecated(json_config):
    """Rename DocModel to DocModel_Deprecated."""
    deprecate(json_config, "DocModel")

    return json_config


@register_adapter(from_version=5)
def old_tasks_deprecated(json_config):
    """
    Rename tasks with data_handler config to _Deprecated
    """
    deprecate(json_config, "BertClassificationTask")
    deprecate(json_config, "BertPairClassificationTask")
    deprecate(json_config, "BertPairwiseClassificationTask")
    deprecate(json_config, "COLMClassifyTask")
    deprecate(json_config, "ContextSCLSTMCompositionalTask")
    deprecate(json_config, "DocClassificationTask")
    deprecate(json_config, "ElmoDocClassificationTask")
    deprecate(json_config, "ElmoFineTunePairwiseClassificationTask")
    deprecate(json_config, "EnsembleTask")
    deprecate(json_config, "FederatedLearningTaskBase")
    deprecate(json_config, "FLDocClassificationTask")
    deprecate(json_config, "FLQueryDocumentPairwiseRankingTask")
    deprecate(json_config, "KDDocClassificationTask")
    deprecate(json_config, "LMTask")
    deprecate(json_config, "PairClassificationTask")
    deprecate(json_config, "PairwiseAttentionClassificationTask")
    deprecate(json_config, "QueryDocumentPairwiseRankingTask")
    deprecate(json_config, "SCLSTMCompositionalTask")
    deprecate(json_config, "SCLSTMTask")
    deprecate(json_config, "SemanticParsingCppTask")
    deprecate(json_config, "SemanticParsingTask")
    deprecate(json_config, "Seq2SeqTask")
    deprecate(json_config, "Seq2SeqCompositionalMetricReporter")
    deprecate(json_config, "Seq2SeqMetricReporter")
    deprecate(json_config, "RNNEncoderDecoder")
    deprecate(json_config, "SeqNNTask")
    deprecate(json_config, "SGNNClassificationTask")
    deprecate(json_config, "ShallowClassificationTask")
    deprecate(json_config, "ShallowTaggingTask")
    deprecate(json_config, "SpanClassificationTask")
    deprecate(json_config, "TreeParserTask")

    return json_config


@register_adapter(from_version=6)
def v6_to_v7(json_config):
    """
    Make `LabelTensorizer` expansible. If the `labels` field should be an instance of
    `LabelTensorizer`, convert it to`{LabelTensorizer: labels}`.
    """
    [(task_name, task)] = json_config["task"].items()
    if task_name in (
        "BertPairRegressionTask",
        "NewDocumentRegression",
        "NewWordTaggingTask",
    ):
        # Task has a label tensorizer different from LabelTensorizer.
        return json_config

    model = task.get("model")
    if not model:
        return json_config

    model_name = None
    if "inputs" in model:
        inputs = model["inputs"]
    elif len(model) == 1:
        [(model_name, model_val)] = model.items()
        inputs = model_val.get("inputs")
    else:
        inputs = None
    if not inputs:
        return json_config
    if model_name in (
        "NewBertRegressionModel",
        "DocRegressionModel",
        "NewWordTaggingModel",
        "ELModel",
        "EntitySalienceModel",
        "MatchaTwoTowerModel",
    ):
        # Model has a label tensorizer different from LabelTensorizer.
        return json_config

    labels = inputs.get("labels")
    if labels is None:
        return json_config

    inputs["labels"] = {"LabelTensorizer": labels}
    return json_config


@register_adapter(from_version=7)
def lm_model_deprecated(json_config):
    """
    Rename LM model to _Deprecated (LMTask is already deprecated in v5)
    """
    deprecate(json_config, "LMLSTM")
    return json_config


@register_adapter(from_version=8)
def new_tasks_rename(json_config):
    """
    Rename tasks with new API consistently
    """
    # Deprecated
    rename(
        json_config,
        "QueryDocumentPairwiseRankingModel",
        "QueryDocumentPairwiseRankingModel_Deprecated",
    )
    # New
    rename(json_config, "NewDocModel", "DocModel")
    rename(json_config, "NewDocRegressionModel", "DocRegressionModel")
    rename(json_config, "NewDocumentClassification", "DocumentClassificationTask")
    rename(json_config, "NewDocumentRegression", "DocumentRegressionTask")
    rename(
        json_config,
        "NewQueryDocumentPairwiseRankingModel",
        "QueryDocPairwiseRankingModel",
    )
    rename(json_config, "NewWordTaggingModel", "WordTaggingModel")
    rename(json_config, "NewWordTaggingTask", "WordTaggingTask")
    rename(json_config, "PairwiseClassification", "PairwiseClassificationTask")
    rename(
        json_config, "QueryDocumentPairwiseRanking", "QueryDocumentPairwiseRankingTask"
    )
    return json_config


@register_adapter(from_version=9)
def move_epoch_size(json_config):
    return rename_parameter(
        json_config, "task.data.epoch_size", "task.trainer.num_batches_per_epoch"
    )


@register_adapter(from_version=10)
def ensemble_task_deprecated(json_config):
    """
    Rename tasks with new API consistently
    """
    # Deprecated
    deprecate(json_config, "BaggingDocEnsemble")
    deprecate(json_config, "BaggingIntentSlotEnsemble")
    deprecate(json_config, "EnsembleTrainer")
    return json_config


@register_adapter(from_version=11)
def rename_bitransformer_inputs(json_config):
    """
    In "BiTransformer" model, rename input "characters" -> "bytes" and update subfields.
    """
    [task] = json_config["task"].values()
    model = task.get("model")

    if model and len(model) == 1 and "BiTransformer" in model:
        model_val = list(model.values())[0]
        if "inputs" not in model_val:
            model_val["inputs"] = {}
        inputs = model_val["inputs"]
        char_config = inputs.pop("characters", {})
        if "max_char_length" in char_config:
            char_config["max_byte_len"] = char_config.pop("max_char_length")
        char_config["offset_for_non_padding"] = 1
        model_val["inputs"]["bytes"] = char_config

    return json_config


@register_adapter(from_version=12)
def v12_to_v13(json_config):
    """remove_output_encoded_layers(json_config)"""
    rename(json_config, "output_encoded_layers", None)
    """
    Make 'ClassificationMetricReporter'
    expansible.

    If the 'metric_reporter' field should be an instance of
    'ClassificationMetricReporter',
    convert it to '{ClassificationMetricReporter: metric_reporter}'.
    """

    [(task_name, task)] = json_config["task"].items()
    if task_name not in (
        "EnsembleTask",
        "DocClassificationTask_Deprecated",
        "DocumentClassificationTask",
        "PairwiseClassificationTask",
        "SeqNNTask",
        "ShallowClassificationTask_Deprecated",
        "KDDocClassificationTask_Deprecated",
        "PairwiseAttentionClassificationTask_Deprecated",
        "ElmoFineTunePairwiseClassificationTask_Deprecated",
        "XLMDocumentClassification",
        "XLMPairClassification",
        "NewBertClassificationTask",
        "NewBertPairClassificationTask",
        "LaserClassificationTask",
    ):
        # Task has a metric reporter different from ClassificationMetricReporter
        return json_config
    metric_reporter = task.get("metric_reporter")
    if metric_reporter is None:
        return json_config
    keys = list(metric_reporter.keys())
    if keys == []:
        return json_config
    set = {"output_path", "model_select_metric", "target_label", "text_column_names"}
    if keys[0] in set:
        task["metric_reporter"] = {"ClassificationMetricReporter": metric_reporter}
    else:
        return json_config
    return json_config


@register_adapter(from_version=13)
def rename_tensorizer_vocab_params(json_config):
    [(task_name, task)] = json_config["task"].items()
    # XLM and Bert models use the `vocab_file` field, but in a custom way. This
    # field should not be migrated to `vocab.vocab_files` as for TokenTensorizer.
    if "XLM" in task_name or "Bert" in task_name:
        return json_config

    def resolve_model(model_config):
        if len(model_config) == 1 and list(model_config)[0][0].isupper():
            [(model_name, model_config)] = model_config.items()
            if "XLM" in model_name or "Bert" in model_name:
                return {}
        return model_config

    model = resolve_model(task.get("model", {}))
    if not model:
        return json_config

    def update_model_config(model_config):
        model_config = resolve_model(model_config)
        tokens = model_config.get("inputs", {}).get("tokens")
        if not tokens:
            return

        vocab = {"build_from_data": tokens.pop("build_vocab", True), "vocab_files": []}
        if "vocab_file" in tokens:
            vocab["vocab_files"].append(
                {
                    "filepath": tokens.pop("vocab_file"),
                    "size_limit": tokens.pop("vocab_file_size_limit", 0),
                }
            )

    if "models" in model:
        # ensemble model
        for sub_model in model["models"]:
            update_model_config(sub_model)
    else:
        update_model_config(model)

    return json_config


@register_adapter(from_version=14)
def flatten_deprecated_ensemble_config(json_config):
    # Deprecated ensemble is removed from codebase, so this is now just a no-op
    return json_config


def migrate_to_new_data_handler(task, columns):
    create_parameter(task, "data.source", {"TSVDataSource": {}})
    rename_parameter(task, "data_handler.eval_path", "data.source.eval_filename")
    rename_parameter(task, "data_handler.test_path", "data.source.test_filename")
    rename_parameter(task, "data_handler.train_path", "data.source.train_filename")
    columns_to_read = next(find_dicts_containing_key(task, "columns_to_read"), None)
    if columns_to_read:
        rename_parameter(
            task, "data_handler.columns_to_read", "data.source.field_names"
        )
    else:
        create_parameter(task, "data.source.field_names", columns)

    rename_parameter(
        task, "data_handler.append_bos", "model.inputs.tokens.add_bos_token"
    )
    rename_parameter(
        task, "data_handler.append_eos", "model.inputs.tokens.add_eos_token"
    )
    rename_parameter(
        task, "data_handler.max_seq_len", "model.inputs.tokens.max_seq_len"
    )

    rename_parameter(
        task, "features.shared_module_key", "model.embedding.shared_module_key"
    )
    rename_parameter(task, "features.word_feat.embed_dim", "model.embedding.embed_dim")
    rename_parameter(task, "features.dense_feat", "model.inputs.dense")

    create_parameter(task, "data.batcher", {"PoolingBatcher": {}})
    rename_parameter(
        task, "data_handler.eval_batch_size", "data.batcher.eval_batch_size"
    )
    rename_parameter(
        task, "data_handler.test_batch_size", "data.batcher.test_batch_size"
    )
    rename_parameter(
        task, "data_handler.train_batch_size", "data.batcher.train_batch_size"
    )

    rename_parameter(
        task,
        "features.word_feat.vocab_size",
        "model.inputs.tokens.vocab.size_from_data",
    )
    rename_parameter(
        task,
        "features.word_feat.vocab_from_train_data",
        "model.inputs.tokens.vocab.build_from_data",
    )

    rename_parameter(
        task,
        "features.word_feat.vocab_file",
        "model.inputs.tokens.vocab.vocab_files",
        lambda x: [{"filepath": x}],
    )

    rename_parameter(task, "labels.label_weights", "model.output_layer.label_weights")

    delete_parameter(task, "data_handler")
    delete_parameter(task, "exporter")
    delete_parameter(task, "features")
    delete_parameter(task, "featurizer")
    delete_parameter(task, "labels")


@register_adapter(from_version=15)
def remove_lmtask_deprecated(json_config):
    for section in find_dicts_containing_key(json_config, "LMTask_Deprecated"):
        task = section.pop("LMTask_Deprecated")
        migrate_to_new_data_handler(task, ["text"])
        section["LMTask"] = task

    return json_config


@register_adapter(from_version=16)
def remove_docclassificationtask_deprecated(json_config):
    for section in find_dicts_containing_key(
        json_config, "DocClassificationTask_Deprecated"
    ):
        task = section.pop("DocClassificationTask_Deprecated")
        convert = next(find_dicts_containing_key(task, "convert_to_bytes"), None)

        section["DocumentClassificationTask"] = task
        migrate_to_new_data_handler(task, ["doc_label", "text"])
        create_parameter(task, "model.inputs.labels.column", "doc_label")

        # In DocumentClassificationTask.Config:
        #   model: BaseModel.Config = DocModel.Config()
        # It will create a BaseModel if model class is implicit in json.
        # We make it explicit to avoid errors.
        for model in find_dicts_containing_key(section, "model"):
            if next(iter(model["model"]))[0].islower():
                model["model"] = {"DocModel": model.pop("model")}

        if convert and convert["convert_to_bytes"]:
            rename(section, "DocModel", "ByteTokensDocumentModel")
    return json_config


@register_adapter(from_version=17)
def rename_fl_task(json_config):
    # remove 'NewDoc' from FL task names
    for trainer_suffix in ["SyncTrainer", "AsyncTrainer"]:
        old_trainer_name = f"FLNewDoc{trainer_suffix}"
        new_trainer_name = f"FL{trainer_suffix}"
        for section in find_dicts_containing_key(json_config, old_trainer_name):
            section[new_trainer_name] = section.pop(old_trainer_name)
    return json_config


@register_adapter(from_version=18)
def upgrade_if_xlm(json_config):
    """
    Make `XLMModel` Union changes for encoder and tokens config.
    Since they are now unions, insert the old class into the config if
    no class name is mentioned.
    """
    _, _, model = find_parameter(json_config, "task.model")
    if model and "XLMModel" in model:
        _, inputs, tokens = find_parameter(json_config, "task.model.inputs.tokens")
        if tokens and "XLMTensorizer" not in tokens:
            inputs["tokens"] = {}
            inputs["tokens"]["XLMTensorizer"] = tokens

    return json_config


@register_adapter(from_version=19)
def fix_fl_local_optimizer_and_trainer(json_config):
    """a) Change FL local optimizer from optimizer:{SGD:{lr=0.1, momentum=0.2}}
    to optimizer:{lr=0.1, momentum=0.2}
    b) Replace trainer:{FLSyncTrainer:{foo}} by
    trainer:{fl_trainer:{foo, type:SyncTrainer}}
    Same for FLAsyncTrainer
    """
    # only for tasks that contain FLSyncTrainer or FLAsyncTrainer
    _, container, trainer = find_parameter(json_config, "task.trainer")
    if not trainer:
        return json_config
    if "FLSyncTrainer" in trainer:
        fl_trainer_name, fl_trainer_type = "FLSyncTrainer", "SyncTrainer"
    elif "FLAsyncTrainer" in trainer:
        fl_trainer_name, fl_trainer_type = "FLAsyncTrainer", "AsyncTrainer"
    else:
        return json_config

    trainer_section = trainer.pop(fl_trainer_name)
    # first, replace optimizer:{SGD:{lr=0.1, momentum=0.2}} by
    # optimizer:{lr=0.1, momentum=0.2}
    if "optimizer" in trainer_section:
        optimizer = trainer_section.pop("optimizer")
        sgd_config = optimizer.pop("SGD")
        trainer_section["optimizer"] = sgd_config
    # rename "global_optimizer" to "aggregator"
    if "global_optimizer" in trainer_section:
        aggregator = trainer_section.pop("global_optimizer")
        trainer_section["aggregator"] = aggregator
    trainer_section["type"] = fl_trainer_type
    trainer["fl_trainer"] = trainer_section
    return json_config


def upgrade_one_version(json_config):
    current_version = json_config.get("version", 0)
    adapter = ADAPTERS.get(current_version)
    if not adapter:
        raise Exception(f"no adapter found for version {current_version}")
    json_config = adapter(json_config)
    eprint(
        f"WARNING - Applying old config adapter for version={current_version}. "
        "Please consider migrating your old configs to the latest version."
    )
    json_config["version"] = current_version + 1
    return json_config


def upgrade_to_latest(json_config):
    current_version = json_config.get("version") or 0
    if current_version > LATEST_VERSION:
        raise Exception(
            f"config version {json_config['version']} shouldn't exceed lastest \
            version {LATEST_VERSION}"
        )
    while current_version != LATEST_VERSION:
        print(f"Current Version: {current_version}")
        json_config = upgrade_one_version(json_config)
        current_version = json_config["version"]
    return json_config
