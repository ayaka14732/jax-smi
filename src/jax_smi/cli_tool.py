import curses
import fire
from glob import glob
import os
import time
import subprocess
from typing import NoReturn

def sanitise_paragraph(p: str) -> str:
    return '\n'.join(line for line in p.split('\n') if line)

def pprof_one_file(filename: str, dir_prefix: str) -> str:
    return filename + '\n' + sanitise_paragraph(subprocess.run(
        args=['go', 'tool', 'pprof', '-tags', os.path.join(dir_prefix, filename)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.decode('utf-8'))

def run(interval: float=1., dir_prefix: str='/dev/shm/jax-smi') -> NoReturn:
    stdscr = curses.initscr()
    try:
        while True:
            stdscr.clear()
            output = '\n\n'.join(pprof_one_file(filepath, dir_prefix=dir_prefix) for filepath in sorted(glob('device-*.prof', root_dir=dir_prefix)))
            stdscr.addstr(output)
            stdscr.refresh()
            time.sleep(interval)
    except KeyboardInterrupt:
        curses.endwin()

def main():
    fire.Fire(run)
