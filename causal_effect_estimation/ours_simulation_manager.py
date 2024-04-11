import os
import subprocess

os.chdir("code")
files_in_directory = os.listdir()
experiments = [f for f in files_in_directory if f.endswith("_ours.py")]

experiments = [f for f in experiments if not f.endswith("BBB_ours.py")]
print(experiments)
for experiment in experiments:
    print(f"Running {experiment}...")
    subprocess.call(['python3', experiment])