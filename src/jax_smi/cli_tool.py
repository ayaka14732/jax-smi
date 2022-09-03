def check_tpu_in_use():
    import fcntl
    from fcntl import LOCK_EX, LOCK_NB
    import posix
    from posix import O_CREAT, O_RDWR
    import sys

    # https://github.com/tensorflow/tensorflow/blob/1a05bad57a2eb870a561daba0255761e0a472d1d/tensorflow/core/tpu/tpu_initializer_helper.cc#L167-L171
    fd = posix.open('/tmp/libtpu_lockfile', O_CREAT | O_RDWR, 0o644)
    try:
        fcntl.lockf(fd, LOCK_EX | LOCK_NB, 0)
    except BlockingIOError:
        sys.exit(-1)

def main():
    import curses
    import time
    import multiprocessing
    import subprocess

    ctx = multiprocessing.get_context('spawn')
    stdscr = curses.initscr()

    def tpu_in_use():
        process = ctx.Process(target=check_tpu_in_use)
        process.start()
        process.join()
        return process.exitcode != 0

    try:
        while True:
            stdscr.clear()
            if not tpu_in_use():
                stdscr.addstr('Waiting for a process to utilise TPU...')
            else:
                output = subprocess.run(
                    args='go tool pprof -tags /dev/shm/memory.prof'.split(' '),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                ).stdout.decode('utf-8')
                stdscr.addstr(output)
            stdscr.refresh()
            time.sleep(1.)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
