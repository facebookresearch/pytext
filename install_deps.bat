@ECHO OFF
python -m pip install --upgrade pip
pip install -e . --process-dependency-links --no-cache-dir --progress-bar off
