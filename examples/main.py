# uncomment the following lines to test on CPU
# import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
# import jax; jax.config.update('jax_platforms', 'cpu')

import functools
import jax
import jax.numpy as np
import random
from jax_smi import initialise_tracking

devices = jax.devices()
n_devices = jax.device_count()

@functools.partial(jax.pmap, axis_name='n_devices')
def some_heavy_computation(a, b):
    c = np.einsum('abcd,ebcd->ae', a, b)
    d = jax.lax.pmean(c, axis_name='n_devices')
    return d

def main():
    initialise_tracking()

    for i in range(1000):
        print(i)

        x = random.randrange(100, 320)
        y = random.randrange(100, 320)

        a = np.zeros((x, 11, 4, 2), dtype=np.float32)
        b = np.zeros((y, 11, 4, 2), dtype=np.float32)

        a = jax.device_put_replicated(a, devices=devices)
        b = jax.device_put_replicated(b, devices=devices)

        some_heavy_computation(a, b)

    print('Done')

if __name__ == '__main__':
    main()
