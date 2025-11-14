import subprocess

# Command as a list of strings
cmd = ["python", "-m", "nuitka", "--mode=app-dist", "--clang", "main.py"]

# Run the command
result = subprocess.run(cmd)