"""Create all features to `features` directory.
"""
from os.path import dirname, basename, isfile, join
import glob
import subprocess
import sys
sys.path.append(".")

if __name__ == "__main__":
    EXCLUDE_FILES = ["__init__.py", "base.py", basename(__file__)]

    modules = glob.glob(join(dirname(__file__), "*.py"))

    features = []
    for module in modules:
        if not isfile(module):
            continue
        if basename(module) in EXCLUDE_FILES:
            continue
        features.append(basename(module)[:-3])

    for feature in features:
        cmd = f"python scripts/features/{feature}.py"
        sts = subprocess.call(cmd, shell=True)
    