"""
Module for making movies
"""
import shutil
import subprocess
from pathlib import Path


def make_movie(moviefiles, filename: Path):
    """
    Make a movie from a set of png files

    :param moviefiles: glob string for png files
    :param filename: output filename
    :return: None
    """

    for out_path in [filename, filename.with_suffix(".gif")]:
        print(f"Saving to: {out_path}")

        ffmpeg_command = f"ffmpeg -an -y -r 20 -i {moviefiles} -b:v 5000k {out_path}"

        subprocess.run(
            ffmpeg_command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    shutil.rmtree(Path(moviefiles).parent)
