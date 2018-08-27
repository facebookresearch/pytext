#!/usr/bin/env python3

from .cli import run_trainer
from .jobspec import register_builtin_jobspecs


register_builtin_jobspecs()

if __name__ == "__main__":
    run_trainer()
