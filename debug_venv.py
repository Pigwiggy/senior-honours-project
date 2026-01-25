import sys, os, platform, pathlib

print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
print("platform:", platform.platform())
print("cwd:", os.getcwd())
print("file:", pathlib.Path(__file__).resolve())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("PATH:", os.environ.get("PATH"))