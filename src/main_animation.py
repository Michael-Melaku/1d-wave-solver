import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Parameters ---
L = 10.0
T = 4.0        # Increased time to see more movement
c = 1.0
nx = 100
dx = L / (nx - 1)
dt = 0.05
nt = int(T / dt)

x = np.linspace(0, L, nx)

# --- 2. Initial Conditions ---
u = np.zeros(nx)
u_prev = np.zeros(nx)
u_next = np.zeros(nx)

# Start with a Gaussian pulse in the middle
u_prev = np.exp(-(x - L/2)**2)
u = np.copy(u_prev)

# --- 3. Setup the Plot ---
fig, ax = plt.subplots()
line, = ax.plot(x, u, color='blue', lw=2)

# Styling the graph
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, L)
ax.set_title("1D Wave Equation Simulation")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Amplitude (u)")
ax.grid(True)

# --- 4. The Update Function ---
# This function runs for every frame of the animation
def update(frame):
    global u, u_prev, u_next
    
    # Finite Difference Calculation (same as before)
    for i in range(1, nx - 1):
        C2 = (c * dt / dx)**2
        u_next[i] = 2*u[i] - u_prev[i] + C2 * (u[i+1] - 2*u[i] + u[i-1])
    
    # Update arrays for the next step
    u_prev = np.copy(u)
    u = np.copy(u_next)
    
    # Update the data in the plot line
    line.set_ydata(u)
    return line,

# --- 5. Create and Save Animation ---
# frames=nt: Run for the total number of time steps calculated
# interval=20: 20 milliseconds delay between frames
anim = FuncAnimation(fig, update, frames=nt, interval=20, blit=True)

print("Rendering animation... please wait.")

# Option A: Show it on screen (Simpler)
#plt.show()

# Option B: Save to file (Requires ffmpeg or pillow)
anim.save('wave_animation.gif', writer='pillow', fps=30)
print("Animation saved as wave_animation.gif")

