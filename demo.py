import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

distance_matrix = np.array([
    [0,  5, 4, 10],
    [5,  0, 8,  5],
    [4,  8, 0,  3],
    [10, 5, 3,  0]
])

permutation_dp, distance_dp = solve_tsp_dynamic_programming(distance_matrix)
permutation_ls, distance_ls = solve_tsp_local_search(distance_matrix)


print(f'Permutation DP: {permutation_dp}\n')
print(f'Distance DP: {distance_dp}\n')

print(f'Permutation Local Search: {permutation_ls}\n')
print(f'Distance Local Search: {distance_ls}\n')


distance_matrix[:, 0] = 0
permutation_open, distance_open = solve_tsp_dynamic_programming(distance_matrix)

print(f'Permutation DP Open: {permutation_open}\n')
print(f'Distance DP Open: {distance_open}\n')


# A more intricate example

from python_tsp.distances import tsplib_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing

# Get corresponding distance matrix
tsplib_file = "tests/tsplib_data/a280.tsp"
distance_matrix = tsplib_distance_matrix(tsplib_file)

# Solve with Local Search using default parameters
permutation_a280, distance_a280 = solve_tsp_local_search(distance_matrix)
# distance: 3064

print(f'Permutation a280: {permutation_a280}\n')
print(f'Distance a280: {distance_a280}\n')

# use Siulated Annealing (SA)
permutation2, distance2 = solve_tsp_simulated_annealing(distance_matrix)

print(f'Permutation a280 (SA): {permutation2}\n')
print(f'Distance a280 (SA): {distance2}\n')

permutation3, distance3 = solve_tsp_local_search(distance_matrix, x0=permutation2)
print(f'Permutation a280 (LS): {permutation3}\n')
print(f'Distance a280 (LS): {distance3}\n')

permutation4, distance4 = solve_tsp_local_search(
    distance_matrix, x0=permutation2, perturbation_scheme="ps3"
)
print(f'Permutation a280 (LS Perturbed): {permutation4}\n')
print(f'Distance a280 (LS Perturbed): {distance4}\n')

# combo
permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
permutation2op, distance2op = solve_tsp_local_search(
    distance_matrix, x0=permutation, perturbation_scheme="ps3"
)
print(f'Permutation a280 (2op): {permutation2op}\n')
print(f'Distance a280 (2op): {distance2op}\n')