import time
import numpy as np

import leap_ec
from leap_ec.problem import ScalarProblem
from leap_ec.algorithm import generational_ea
from leap_ec.representation import Representation
import leap_ec.ops as ops

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip

from leap_ec.real_rep.problems import SpheroidProblem, WeierstrassProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian

from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint


class MaxN(ScalarProblem):
    def __init__(self, n, maximize=True):
        super().__init__(maximize=maximize)
        self.n = n

    def evaluate(self, phenome):
        return np.count_nonzero(phenome == self.n)

print(f'LEAP version: {leap_ec.__version__}')

POP_SIZE = 10
GENOME_LENGTH = 100
MAX_GEN = 100

BIN_PROBLEM = MaxOnes()
REAL_PROBLEM = WeierstrassProblem()
INT_PROBLEM = MaxN(5)

# ================= Binary Generational EA ================================
bin_start = time.time()
bin_ea = generational_ea(max_generations=MAX_GEN, pop_size=POP_SIZE,
                         problem=BIN_PROBLEM,
                         representation=Representation(
                             initialize=create_binary_sequence(GENOME_LENGTH)),
                         pipeline=[
                             ops.tournament_selection(k=2),
                             ops.clone,
                             mutate_bitflip(probability=0.5),
                             ops.evaluate,
                             ops.pool(size=POP_SIZE)
                         ])
list(bin_ea)
bin_elapsed = time.time() - bin_start
# =========================================================================


# ================= Real Generational EA ================================
real_start = time.time()
real_ea = generational_ea(max_generations=MAX_GEN, pop_size=POP_SIZE,
                          problem=REAL_PROBLEM,
                         representation=Representation(
                             initialize=create_real_vector(
                                 bounds=[REAL_PROBLEM.bounds]*GENOME_LENGTH)),
                         pipeline=[
                             ops.tournament_selection(k=2),
                             ops.clone,
                             mutate_gaussian(std=0.5, expected_num_mutations='isotropic'),
                             ops.evaluate,
                             ops.pool(size=POP_SIZE)
                         ])
list(real_ea)
real_elapsed = time.time() - real_start
# =========================================================================


# ================= Integer Generational EA ================================
int_start = time.time()
int_ea = generational_ea(max_generations=MAX_GEN, pop_size=POP_SIZE,
                         problem=INT_PROBLEM,
                         representation=Representation(
                             initialize=create_int_vector(
                                 bounds=[[0, 10]]*GENOME_LENGTH)),
                         pipeline=[
                             ops.tournament_selection(k=2),
                             ops.clone,
                             mutate_randint(bounds=[[0, 10]]*GENOME_LENGTH,
                                                        probability=0.5),
                             ops.evaluate,
                             ops.pool(size=POP_SIZE)
                         ])
list(int_ea)
int_elapsed = time.time() - int_start
# =========================================================================

print(f'MaxOnes generational runtime: {bin_elapsed:.3f}s')
print(f'WeierstrassProblem generational runtime: {real_elapsed:.3f}s')
print(f'MaxN generational runtime: {int_elapsed:.3f}s')

