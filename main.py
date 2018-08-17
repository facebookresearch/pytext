#!/usr/bin/env python3

import libfb.py.fbpkg as fbpkg
from pytext.common.registry import register_jobspec
from pytext.config import PyTextConfig
from pytext.fb.experimental import I18NDocClassifyJobSpec, I18NJointTextJobSpec
from pytext.workflow import test_model, train_model

from .args import parse_config
from .jobspec import register_builtin_jobspecs


register_builtin_jobspecs()
register_jobspec([I18NJointTextJobSpec, I18NDocClassifyJobSpec])


def main():
    config = parse_config()
    if config.test_given_snapshot:
        test_model(config)
    else:
        if config.jobspec.data_handler.pretrained_embeds_file:
            print("Fetching embedding pkg")
            pretrained_embeds_file = fbpkg.fetch(
                config.jobspec.data_handler.pretrained_embeds_file,
                dst="/tmp",
                verbose=False,
            )
            config = config._replace(
                jobspec=config.jobspec._replace(
                    data_handler=config.jobspec.data_handler._replace(
                        pretrained_embeds_file=pretrained_embeds_file
                    )
                )
            )
        print("Starting training...")
        train_model(config)
        print("Starting testing...")
        test_config_dict = config._asdict()
        test_config_dict["load_snapshot_path"] = config.save_snapshot_path
        test_config = PyTextConfig(**test_config_dict)
        test_model(test_config)


if __name__ == "__main__":
    main()
