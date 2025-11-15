import subprocess

subprocess.run(['python', "-m", 'nuitka', '--follow-imports', "--include-data-dir=train_assets=./train_assets", "--include-data-dir=eval_assets=./eval_assets","--mode=app-dist", "--clang", "--enable-plugins=matplotlib", 'main.py'])