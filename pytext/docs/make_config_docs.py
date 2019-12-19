#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import contextlib
import itertools
import json
import os
import typing

from pytext.config import ConfigBase, PyTextConfig
from pytext.config.component import Registry
from pytext.config.serialize import config_to_json
from pytext.utils.file_io import PathManager
from sphinx.ext.napoleon import GoogleDocstring
from sphinx.pycode import ModuleAnalyzer


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "source/configs")


def canonical_path(config):
    module = config.__module__
    if "__COMPONENT__" in vars(config):
        module = config.__COMPONENT__.__module__
    return f"{module}.{config.__name__}"


class Config(typing.NamedTuple):
    name: str
    path: str
    config: ConfigBase

    @classmethod
    def from_config(cls, config: ConfigBase):
        return cls(config.__name__, canonical_path(config), config)


class ConfigReference(typing.NamedTuple):
    path: str


def find_additional_configs(configs, seen=None):
    if seen is None:
        seen = set()

    for config in configs:
        if config in seen:
            continue
        elif hasattr(config, "__args__"):
            yield from find_additional_configs(config.__args__, seen)
            continue
        elif not isinstance(config, type) or not issubclass(config, ConfigBase):
            continue

        seen.add(config)
        yield config
        annotations, defaults = config.annotations_and_defaults()
        yield from find_additional_configs(annotations.values(), seen)


ALL_CONFIG_CLASSES = find_additional_configs(
    itertools.chain.from_iterable(Registry._registered_components.values())
)

CONFIG_WRAPPERS = [Config.from_config(config) for config in ALL_CONFIG_CLASSES]

assert len({config.path for config in CONFIG_WRAPPERS}) == len(
    CONFIG_WRAPPERS
), "Some configs had exactly identical config paths!"

ALL_CONFIGS = {config.path: config for config in CONFIG_WRAPPERS}
ALL_CONFIGS[canonical_path(PyTextConfig)] = Config.from_config(PyTextConfig)

NO_DEFAULT = object()


def marked_up_type_name(arg_type):
    if getattr(arg_type, "__origin__", None) is typing.Union:
        if len(arg_type.__args__) == 2 and type(None) in arg_type.__args__:
            optional_type = next(
                t for t in arg_type.__args__ if not isinstance(t, type(None))
            )
            return f"Optional[{marked_up_type_name(optional_type)}]"
        options = [marked_up_type_name(t) for t in arg_type.__args__]
        return f"Union[{', '.join(options)}]"
    elif hasattr(arg_type, "__args__"):
        options = [marked_up_type_name(t) for t in arg_type.__args__]
        type_name = (
            arg_type.__origin__.__name__ if arg_type.__origin__ else arg_type.__name__
        )
        return f"{type_name}[{', '.join(options)}]"
    elif arg_type is typing.Any:
        return f"Any"
    elif issubclass(arg_type, ConfigBase):
        path = canonical_path(arg_type)
        name = arg_type.__name__
        return f":doc:`{name} <{path}>`"
    else:
        return arg_type.__name__


def marked_up_default_value(arg_value):
    if isinstance(arg_value, ConfigBase):
        annotations, defaults = type(arg_value).annotations_and_defaults()
        values = {
            name: value
            for name, value in vars(arg_value).items()
            if name in annotations and value != defaults.get(name)
        }
        path = canonical_path(type(arg_value))
        values_str = ", ".join(
            f"{name}=\\ {marked_up_default_value(value)}\\ "
            for name, value in values.items()
        )
        path = canonical_path(type(arg_value))
        name = type(arg_value).__name__
        return f":doc:`{name} <{path}>`\\ ({values_str})"

    return f"``{repr(arg_value)}``"


class Attribute(typing.NamedTuple):
    type: typing.Type
    name: str
    default: typing.Any
    docstring: str

    def __repr__(self):
        type = marked_up_type_name(self.type)
        default_value = marked_up_default_value(self.default)
        return f"-- {self.name}: {type} = {default_value} {self.docstring}"


def config_attrs(config):
    annotations, defaults = config.config.annotations_and_defaults()
    analyzer = ModuleAnalyzer.for_module(config.config.__module__)
    attr_docs = {
        attr: list(lines)
        for (classname, attr), lines in analyzer.find_attr_docs().items()
        if classname == config.name
    }
    return [
        Attribute(type, name, defaults.get(name, NO_DEFAULT), attr_docs.get(name, ""))
        for name, type in annotations.items()
    ]


def I(n):
    return "  " * n


def unindent_docstring(docstring):
    lines = docstring.splitlines()
    first, *rest = lines or [""]
    second = next((line for line in rest if line), "")
    indent = len(second) - len(second.lstrip())
    if any(line[:indent].strip() for line in rest):
        raise ValueError("Unexpected unindent in docstring")
    return [first] + [line[indent:] for line in rest]


def rst_big_header(s):
    return f"{s}\n{'='*len(s)}\n"


def rst_little_header(s):
    return f"{s}\n{'='*len(s)}\n"


def rst_toctree(elements):
    return (
        "\n\n".join(
            (".. toctree::", "\n".join(I(1) + e for e in sorted(elements) if e))
        )
        if elements
        else ""
    )


def subclass_configs(config):
    for child in ALL_CONFIGS.values():
        if child.config is not config and issubclass(child.config, config):
            yield child


def format_config_rst(config):
    attrs = config_attrs(config)

    def join(elements):
        return "\n\n\n".join(e for e in elements if e)

    bases = [Config.from_config(base) for base in config.config.__bases__]

    config_doc = "\n".join(
        (
            f".. py:currentmodule:: {config.config.__module__}",
            f".. py:class:: {config.config.__name__}",
            I(1) + ":noindex:",
            "",
            I(1)
            + "**Bases:** "
            + ", ".join(f":class:`{base.name} <{base.path}>`\\ " for base in bases),
            "",
            *(
                I(1) + line
                for line in GoogleDocstring(
                    unindent_docstring(config.config.__doc__ or "")
                ).lines()
            ),
            "",
            "**All Attributes (including base classes)**",
            "",
            *itertools.chain.from_iterable(
                (
                    I(1)
                    + f"**{attr.name}**: {marked_up_type_name(attr.type)}"
                    + (
                        f" = {marked_up_default_value(attr.default)}"
                        if attr.default is not NO_DEFAULT
                        else ""
                    ),
                    *(I(2) + line for line in attr.docstring or ("\\ ",)),
                    "",
                )
                for attr in attrs
            ),
        )
    )

    try:
        config_json = json.dumps(
            config_to_json(config.config, config.config()), indent=4
        )
    except Exception as e:
        print(e)
        config_json = ""

    subclasses = sorted(subclass_configs(config.config), key=lambda c: c.path)

    return join(
        (
            rst_big_header(config.config.__name__),
            *(
                (
                    f"**Component:** :class:`{config.config.__COMPONENT__.__name__} "
                    + f" <{canonical_path(config.config.__COMPONENT__)}>`\ ",
                )
                if hasattr(config.config, "__COMPONENT__")
                else ()
            ),
            config_doc,
            "\n".join(
                (
                    "**Subclasses**",
                    *(
                        I(1) + f"- :class:`{child.name} <{child.path}>`\\ "
                        for child in subclasses
                    ),
                )
            )
            if hasattr(config.config, "__EXPANSIBLE__") and subclasses
            else "",
            *(
                (
                    "**Default JSON**",
                    ".. code-block:: json",
                    "\n".join(I(1) + line for line in config_json.split("\n")),
                )
                if config_json
                else (
                    "\n".join(
                        (
                            ".. warning::",
                            I(1) + "This config has parameters with no default values.",
                            I(1)
                            + "We aren't yet able to generate functional JSON for it.",
                        )
                    ),
                )
            ),
        )
    )


def toctree_files(config_paths):
    # The first element is subpackages, the second is configs.
    packages = collections.defaultdict(lambda: (set(), set()))
    for path in config_paths:
        if path.endswith(".Config"):
            package, name, _ = path.rsplit(".", 2)
            name = f"{name}.Config"
        else:
            package, name = path.rsplit(".", 1)
        packages[package][1].add(path)
        while "." in package:
            base_package, subpackage = package.rsplit(".", 1)
            packages[base_package][0].add(package)
            package = base_package

    for package, (subpackages, configs) in packages.items():
        yield (package, rst_toctree(subpackages), rst_toctree(configs))


def format_toctree_rst(package, subpackages, configs):
    return "\n\n".join(
        (
            rst_big_header(package.split(".")[-1]),
            *((subpackages,) if subpackages else ()),
            *((configs,) if configs else ()),
        )
    )


def main():
    with contextlib.suppress(FileExistsError):
        os.mkdir(CONFIG_DIR)

    common_prefix = os.path.commonprefix((__file__, CONFIG_DIR))

    for config_path, config in ALL_CONFIGS.items():
        file_path = os.path.join(CONFIG_DIR, f"{config_path}.rst")
        print("Creating file", os.path.relpath(file_path, common_prefix))
        with PathManager.open(file_path, "w") as config_rst:
            config_rst.write(format_config_rst(config))

    for package, subpackages, configs in toctree_files(ALL_CONFIGS):
        file_path = os.path.join(CONFIG_DIR, f"{package}.rst")
        print("Creating file", os.path.relpath(file_path, common_prefix))
        with PathManager.open(file_path, "w") as toctree_file:
            toctree_file.write(format_toctree_rst(package, subpackages, configs))


if __name__ == "__main__":
    main()
