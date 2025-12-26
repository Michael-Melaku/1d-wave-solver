import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0  # Length of the string
T = 2.0   # Total time
c = 1.0   # Wave speed
nx = 100  # Number of grid points
dx = L / (nx - 1)
dt = 0.05 # Time step (ensure Courant condition c*dt/dx <= 1)
nt = int(T / dt)

# Arrays
u = np.zeros(nx)      # Current step
u_prev = np.zeros(nx) # Previous step
u_next = np.zeros(nx) # Next step
x = np.linspace(0, L, nx)

# Initial Condition (Gaussian pulse)
u_prev = np.exp(-(x - L/2)**2)
u = np.copy(u_prev) # Assume zero initial velocity

# Time stepping loop
for n in range(nt):
    for i in range(1, nx - 1):
        u_next[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
    
    # Update arrays
    u_prev = np.copy(u)
    u = np.copy(u_next)

# Plot final state
plt.plot(x, u)
plt.title(f"Wave at t={T}")
plt.xlabel("Position (x)")
plt.ylabel("Amplitude (u)")
plt.savefig("final_wave.png")
plt.show()

