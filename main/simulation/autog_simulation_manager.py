import os
import subprocess

files_in_directory = os.listdir()
experiments = [f for f in files_in_directory if f.endswith("_autog.py")]

for experiment in experiments:
    print(f"Running {experiment}...")
    subprocess.call(['python3', experiment])