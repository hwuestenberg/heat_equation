#####################################################################
# Created by strider
# at 21/12/2020

# Feature: Stationary heat equation solver
# Solve the stationary heat equation
#####################################################################


## IMPORT
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt



## BODY

# Define domain (1D rod)
xmin = 0
xmax = 1
xlen = abs(xmax - xmin)
X = np.linspace(xmin, xmax, 100)
dx = abs(X[1] - X[0])
N = X.size
print(f"X step size:\t{dx}")

# Define material properties
k = 1  # Conductivity
A_factor = k/dx**2

# Define toeplitz column
A_col = [2, -1] + [0] * (N - 2)

# Define boundary conditions
bc_left = 100
bc_right = 0

# Define right-hand side
h = 0
H_vector = [h + A_factor * bc_left] + [h] * (N - 2) + [h + A_factor * bc_right]

# Define tensors
A = A_factor * linalg.toeplitz(A_col)  # N x N matrix
T = np.zeros(N)  # vector with N entries
H = np.array(H_vector)  # vector with N entries

# Solve linear system
stat_T = linalg.solve(A, H)  # solution for BCs

# Get analytical solution
exact_T = -(bc_left - bc_right) / xlen * X + bc_left

# Compare temperature
err = linalg.norm(exact_T - stat_T, 2)
max_err = np.max(np.abs(exact_T - stat_T))
min_err = np.min(np.abs(exact_T - stat_T))
print(f"Absolute error:\t{err}")
print(f"Max error:\t{max_err}")
print(f"Min error:\t{min_err}")

# Show temperature
fig, ax = plt.subplots()
ax.plot(X, exact_T, 'b-', label="exact")
ax.plot(X, stat_T, 'b--', label="stationary")
ax.legend()
plt.show()
