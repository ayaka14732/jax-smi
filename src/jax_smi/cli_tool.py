import subprocess
import time

import fire

from .common_utils import ON_TPU

def print_info_tpu(interval: float=1.) -> None:
    import rich.console
    from tpu_info.cli import print_chip_info

    console = rich.console.Console()

    while True:
        console.clear()
        print_chip_info()
        time.sleep(interval)

def print_info_non_tpu(interval: float=1., dir_prefix: str='/dev/shm') -> None:
    import curses

    stdscr = curses.initscr()
    try:
        while True:
            stdscr.clear()
            output = subprocess.run(
                args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.decode('utf-8')
            stdscr.addstr(output)
            stdscr.refresh()
            time.sleep(interval)
    except KeyboardInterrupt:
        curses.endwin()

def run(interval: float=1., dir_prefix: str='/dev/shm'):
    if ON_TPU:
        print_info_tpu()
    else:
        print_info_non_tpu(interval=interval, dir_prefix=dir_prefix)

def main():
    fire.Fire(run)
