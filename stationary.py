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
X = np.linspace(xmin, xmax, 100)
dim = X.size

# Define material properties
rho = 1  # Density [kg/m^3]
cp = 1  # Specific heat capacity for const pressure
k = 1  # Conductivity
alpha = k / (rho * cp)  # Thermal diffusivity

# Define toeplitz column
A_row = [2, -1] + [0] * (dim-2)

# Define boundary conditions
bc_left = 100
bc_right = 0

# Define right-hand side
h = 0
B_vector = [bc_left + h] + [h] * (dim-2) + [bc_right + h]

# Define tensors
A = linalg.toeplitz(A_row)  # dim x dim matrix
T = np.zeros(dim)  # vector with dim entries
B = np.array(B_vector)  # vector with dim entries

# Solve linear system
T = linalg.solve(A, B)  # solution for BCs

# Show T
fig, ax = plt.subplots()
ax.plot(X, T)
plt.show()