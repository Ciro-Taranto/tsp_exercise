import numpy as np
from time import time

for e in range(7):
    for _ in range(10):
        cs_1 = set(np.random.randint(low=0, high=1e9, size=10 ** e))
        cs_2 = set(np.random.randint(low=0, high=1e9, size=10 ** e))

        t0 = time()
        cs_1.copy().update(cs_2)
        print(f'union: size: {10 ** e}, time: {time() - t0}')

        cs_1_list = list(cs_1)
        cs_2_list = list(cs_2)

        t0 = time()
        cs_1_list.extend(cs_2_list)
        print(f'union list: size: {10 ** e}, time: {time() - t0}')
