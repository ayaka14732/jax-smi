import curses
import fire
import time
import subprocess

def run(interval: float=1., dir_prefix: str='/dev/shm'):
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

def main():
    fire.Fire(run)
