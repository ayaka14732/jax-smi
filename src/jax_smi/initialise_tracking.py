from .common_utils import ON_TPU

def initialise_tracking(interval: float=1., dir_prefix: str='/dev/shm') -> None:
    # on TPU, we use `tpu-info` directly, so nothing needs to be done here
    if ON_TPU:
        return

    import jax
    import threading

    def inner():
        import posix
        import time
        while True:
            jax.profiler.save_device_memory_profile(f'{dir_prefix}/memory.prof.new')
            posix.rename(f'{dir_prefix}/memory.prof.new', f'{dir_prefix}/memory.prof')  # atomic
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()
