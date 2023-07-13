def initialise_tracking(interval: float=1., dir_prefix: str='/dev/shm/jax-smi', rank: str=0) -> None:
    import jax
    import os
    import threading

    os.makedirs(dir_prefix, exist_ok=True)

    filepath = os.path.join(dir_prefix, f'device-{rank}.prof')
    filepath_new = f'{filepath}.new'

    def inner():
        import posix
        import time
        while True:
            jax.profiler.save_device_memory_profile(filepath_new)
            posix.rename(filepath_new, filepath)  # atomic
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()
