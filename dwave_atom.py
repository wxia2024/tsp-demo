import time
import dwave.system
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from dwave.system import LeapHybridSampler

# Define the QUBO matrix (this is a simplified example for illustration)
#Q = np.array([[1, -1], [-1, 2]])

# Define a more realistic QUBO with more complex interactions (simplified)
Q = np.zeros((4, 4))
Q[0, 1] = Q[1, 0] = -2
Q[1, 2] = Q[2, 1] = -2
Q[2, 3] = Q[3, 2] = -2
Q[0, 3] = Q[3, 0] = 6
print(Q)
# # # Convert Q into a dictionary format suitable for D-Wave
Q_dict = {(i, j): Q[i, j] for i in range(len(Q)) for j in range(i, len(Q))}
print("")
print(Q_dict)
# Set up D-Wave sampler (assuming you have access to a D-Wave system)
sampler = EmbeddingComposite(DWaveSampler())
#sampler = EmbeddingComposite(LeapHybridSampler())

# Solve the problem
response = sampler.sample_qubo(Q_dict, num_reads=4000)

# Ensure response.samples() gives all the samples
samples = list(response.samples())  # This will convert the generator to a list
print("")
print(samples)
# # # Access energies from response.data() (this returns a list of named tuples)
energies = list(response.data(fields=['energy']))

# # Iterate through both samples and energies
for sample, energy in zip(samples, energies):
    print(f"Sample: {sample}, Energy: {energy[0]}")
