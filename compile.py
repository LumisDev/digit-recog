import subprocess

# Command as a list of strings
cmd = ["python", "-m", "nuitka", "--mode=app-dist", "--clang", "--verbose", "--show-scons", "main.py"]

# Run the command
result = subprocess.run(cmd)