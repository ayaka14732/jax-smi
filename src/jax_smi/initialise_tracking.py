def initialise_tracking():
    import jax
    import threading

    def inner():
        import posix
        import time
        while True:
            jax.profiler.save_device_memory_profile('/dev/shm/memory.prof.new')
            posix.rename('/dev/shm/memory.prof.new', '/dev/shm/memory.prof')  # atomic
            time.sleep(1.)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()
