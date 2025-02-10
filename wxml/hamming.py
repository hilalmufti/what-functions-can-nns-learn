import random
from functools import partial

import torch as th
from toolz.curried import *


def to_base(x, b):
    def go(x, b, acc):
        if x == 0:
            return list(reversed(acc))
        else:
            acc.append(x % b)
            return go(x // b, b, acc)
    return go(x, b, [])


def bits(x):
    return to_base(x, 2)


def bitvector(x, n):
    def go(n, acc):
        if n == 0:
            return acc
        else:
            acc.insert(0, 0)
            return go(n - 1, acc)
        
    acc = bits(x)
    assert n - len(acc) >= 0, f"n={n} is too small for x={x}"
    return go(n - len(acc), acc)
bitvector8 = partial(bitvector, n=8)
bitvector16 = partial(bitvector, n=16)
bitvector32 = partial(bitvector, n=32)
bitvector64 = partial(bitvector, n=64)


def bittensor(x, n):
    return th.tensor(bitvector(x, n))
bittensor8 = partial(bittensor, n=8)
bittensor16 = partial(bittensor, n=16)
bittensor32 = partial(bittensor, n=32)
bittensor64 = partial(bittensor, n=64)

# hamming weight
def wt(xs):
    return th.sum(xs, dim=-1)


def make_data_hamming(n):
    xs = list(map(bittensor8, take(n, iterate(lambda _: random.randint(0, 64), random.randint(0, 64)))))
    ys = list(map(compose(bittensor8, wt), xs))
    return xs, ys
