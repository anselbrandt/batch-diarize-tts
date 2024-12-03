import os
import subprocess
import time

ROOT = os.getcwd()
audioDir = os.path.join(ROOT, "clean")
outputDir = os.path.join(ROOT, "output")

os.makedirs(outputDir, exist_ok=True)


def getFiles(dir):
    dirs = [(os.path.join(dir, subdir), subdir) for subdir in os.listdir(dir)]

    files = [
        (os.path.join(subdir, file), dir, file)
        for subdir, dir in dirs
        for file in os.listdir(subdir)
    ]
    return sorted(files)


def getFilenames(dir):
    dirs = [(os.path.join(dir, subdir), subdir) for subdir in os.listdir(dir)]

    files = [
        os.path.join(dir, subdir, file)
        for subdir, dir in dirs
        for file in os.listdir(subdir)
    ]
    return sorted(files)


files = getFiles(audioDir)

for file in files:
    start_time = time.time()
    completed = getFilenames(outputDir)
    filepath, showname, filename = file
    output_dir = os.path.join(ROOT, "output", showname)
    output_filename = os.path.join(ROOT, output_dir, filename.replace(".wav", ".srt"))
    if output_filename not in completed:
        subprocess.run(
            [
                "python",
                "diarize.py",
                "-a",
                filepath,
            ]
        )
