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
