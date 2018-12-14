@ECHO OFF
::Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
::Use venv name if passed, otherwise default
IF "%1"=="" (
  SET "_PYTEXT_ENV_NAME_=pytext_venv"
) ELSE (
  SET "_PYTEXT_ENV_NAME_=%1"
)

IF NOT EXIST %_PYTEXT_ENV_NAME_% (
  python -m venv %_PYTEXT_ENV_NAME_%
)

call %_PYTEXT_ENV_NAME_%\Scripts\activate.bat
