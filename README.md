# JAX Synergistic Memory Inspector

![](https://raw.githubusercontent.com/ayaka14732/jax-smi/main/demo/1.gif)

`jax-smi` is a tool for real-time inspection of the memory usage of a JAX process. It is similar to `nvidia-smi` for GPU, but works on multiple platforms including CPU, GPU and TPU.

On TPU platforms, `jax-smi` is the only way to monitor TPU memory usage. On GPU platforms, `jax-smi` is also preferable to `nvidia-smi`. The latter is unable to report real-time memory usage of JAX processes, as JAX always [pre-allocates 90% of the GPU memory](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) by default.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Installation

Install `go`. On Ubuntu, this is usually done by:

```sh
sudo apt-get install golang
```

If you followed [tpu-starter](https://github.com/ayaka14732/tpu-starter) to set up the TPU environment, `go` should be already installed.

Then install `jax-smi` with:

```sh
pip install jax-smi
```

## Usage

In your JAX script:

```python
from jax_smi import initialise_tracking
initialise_tracking()
# some computation...
```

Open a shell and run:

```sh
jax-smi
```

## Approach

Update: Since v2.0.0, `jax-smi` calls `tpu-info` directly on TPU platforms.

Save the memory profile to `/dev/shm/memory.prof` in a separate thread every 1 second using [`jax.profiler.save_device_memory_profile()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.save_device_memory_profile.html).

Inspect the memory profile with `go tool pprof -tags /dev/shm/memory.prof`.

See <https://twitter.com/ayaka14732/status/1565013139594551296> for more details.

## Limitations

Tracing can only be performed by one process at a time. If tracing is performed by multiple JAX processes, they will write the memory profiles to the same file, which will lead to conflicts.

The `jax-smi` command line tool cannot detect if a memory profile file is out of date. Therefore, even if no JAX process is running, the tool will still read the outdated memory profile and report outdated memory usage information.
