import os
import subprocess

# Get a list of all files in current directory
files_in_directory = os.listdir()

# Filter out all files that are not .py files
experiment_files = [f for f in files_in_directory if f.endswith("_autog.py") or 
                    f.endswith("_ours.py")]

# Run each experiment file
for experiment_file in experiment_files:
    print(f"Running {experiment_file}...")
    subprocess.call(['python3', experiment_file])