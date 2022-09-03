def initialise_tracking(interval: float=1.) -> None:
    import jax
    import threading

    def inner():
        import posix
        import time
        while True:
            jax.profiler.save_device_memory_profile('/dev/shm/memory.prof.new')
            posix.rename('/dev/shm/memory.prof.new', '/dev/shm/memory.prof')  # atomic
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()
