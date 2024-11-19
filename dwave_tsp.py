
import time
from dwave.optimization.generators import traveling_salesperson
from dwave.system import LeapHybridNLSampler
from python_tsp.distances import tsplib_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

# Instructions:
# 
#  1. Decide if you want to use small, medium, or large matrix, and uncomment that one. Leave rest commented.
#  2. Run the file, it will solve the problem using quantum computer first, then normal computer local search, the normal computer dynamic programming
#
#  NOTE: for large matrix, the normal computer, especially dynamic programming, may take a long time, 10min to 1 hour

#### Use small matrix n = 5 ####

# DISTANCE_MATRIX = [
#     [0, 656, 227, 578, 489],
#     [656, 0, 889, 141, 170],
#     [227, 889, 0, 773, 705],
#     [578, 141, 773, 0, 161],
#     [489, 170, 705, 161, 0]]

#### Use medium matrix n = 280 ####

# tsplib_file = "tests/tsplib_data/a280.tsp"
# DISTANCE_MATRIX = tsplib_distance_matrix(tsplib_file)

#### Use large matrix ####

# Define the size of the matrix
size = 100
# Create a random matrix
#A = np.random.rand(size, size)
A = np.random.randint(size, size)
# Make the matrix symmetric by setting A[i, j] = A[j, i]
A = (A + A.T) / 2
np.fill_diagonal(A, 0)
# Optionally, print a portion of the matrix to verify
# (printing the whole 10k x 10k matrix is impractical)
print(A[:5, :5])  # Print a 5x5 portion of the matrix
DISTANCE_MATRIX = A

#### Running Quantum Computer ####
start_time = time.time()
model = traveling_salesperson(distance_matrix=DISTANCE_MATRIX)
sampler = LeapHybridNLSampler()                  
results = sampler.sample(
    model,
    label='SDK Examples - TSP')     
route, = model.iter_decisions()  
# print the answer
print(route.state(0))  

dwave_time = time.time() - start_time
print(f"Dwave operation took {dwave_time:.4f} seconds")

#### Running Normal Computer Local Search #### 
distance_matrix = np.array(DISTANCE_MATRIX)

start_time = time.time()
permutation_ls, distance_ls = solve_tsp_local_search(distance_matrix)
print(f'Permutation Local Search: {permutation_ls}\n')
print(f'Distance Local Search: {distance_ls}\n')
normal_ls_time = time.time() - start_time
print(f"Normal Computer local_search operation took {normal_ls_time:.4f} seconds")

### Running Normal Computer Dynamic Programming ####
start_time = time.time()
permutation_dp, distance_dp = solve_tsp_dynamic_programming(distance_matrix)
print(f'Permutation DP: {permutation_dp}\n')
print(f'Distance DP: {distance_dp}\n')
normal_dp_time = time.time() - start_time
print(f"Normal Computer dynamic_programming operation took {normal_dp_time:.4f} seconds")
