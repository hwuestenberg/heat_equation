#####################################################################
# Created by strider
# at 21/12/2020

# Feature: Instationary heat equation solver
# Solve the instationary heat equation
#####################################################################


## IMPORT
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt



## BODY
# Define solver parameters
omega = 1
tol = 1e-4

# Define domain (1D rod)
xmin = 0
xmax = 1
xlen = abs(xmax - xmin)
X = np.linspace(xmin, xmax, 100)
dx = abs(X[1] - X[0])
N = X.size

# Define time frame
tmin = 0
tmax = 1
t = np.linspace(tmin, tmax, 20000)
dt = abs(t[1] - t[0])
M = t.size

# Define material properties
rho = 1  # Density [kg/m^3]
cp = 1  # Specific heat capacity for const pressure
k = 1  # Conductivity
alpha = k / (rho * cp)  # Thermal diffusivity
A_factor = alpha / dx**2

# Check stability
stability_condition = A_factor * dt
if stability_condition < 1/2:
    print("Method is stable and positive")
if stability_condition < 1/4:
    print("Non-oscillatory")

# Define toeplitz column
A_col = [2, -1] + [0] * (N - 2)

# Define boundary conditions
bc_left = 100
bc_right = 0

# Define right-hand side
h = 0
H_vector = [A_factor * bc_left + h] + [h] * (N - 2) + [A_factor * bc_right + h]

# Define initial conditions
ic = [0] * N

# Define tensors
A = A_factor * linalg.toeplitz(A_col)  # N x N matrix
T = np.zeros((M, N))  # vector with N entries for each time step
T[0, :] = np.array(ic)  # set initial conditions
H = np.array(H_vector)  # vector with N entries

# Solve stationary solution
stat_T = linalg.solve(A, H)  # stationary solution

# Get analytical solution
exact_T = -(bc_left - bc_right) / xlen * X + bc_left

# Solve instationary solution
## Richardson
for i in range(1, M):
    T[i] = T[i-1] + omega * dt * (H - A.dot(T[i-1]))  # step

    # Check error
    # err = linalg.norm(T[i+1] - T[i], 2) / linalg.norm(T[i], 2)
    # if err > tol:
    #     print(f"Error smaller than tolerance:\t{err.__round__(10)} < {tol.__round__(10)}")
    #     break

# Compare
fig, ax = plt.subplots()
ax.plot(X, exact_T, label="exact")
ax.plot(X, stat_T, label="stationary")
ax.plot(X, T[-1], label="instationary")
ax.legend()
plt.show()
