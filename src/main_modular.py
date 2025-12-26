import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Configuration & Initialization ---

def get_parameters():
    """
    Returns a dictionary containing physical and simulation parameters.
    """
    params = {
        'L': 10.0,       # Length of the domain
        'T': 4.0,        # Total simulation time
        'c': 1.0,        # Wave speed
        'nx': 100,       # Number of spatial grid points
        'dt': 0.05,      # Time step
    }
    # Derived parameter: dx
    params['dx'] = params['L'] / (params['nx'] - 1)
    
    # Check Courant condition for stability
    C = params['c'] * params['dt'] / params['dx']
    if C > 1:
        print(f"WARNING: Courant number is {C:.2f} (> 1). Simulation may be unstable.")
    
    return params

def initialize_fields(params):
    """
    Initializes the wave field arrays u, u_prev, and u_next.
    Sets the initial condition (Gaussian pulse).
    """
    nx = params['nx']
    dx = params['dx']
    L = params['L']
    
    # Create spatial grid
    x = np.linspace(0, L, nx)
    
    # Initialize arrays
    u = np.zeros(nx)
    u_prev = np.zeros(nx)
    u_next = np.zeros(nx)
    
    # Initial Condition: Gaussian Pulse
    # Centered at L/2 with a specific width
    u_prev = np.exp(-(x - L/2)**2)
    
    # Initial Velocity Condition:
    # Assuming initial velocity is zero, u at t=1 is roughly same as u at t=0
    # (For higher precision, one would use a specific starting formula here)
    u = np.copy(u_prev)
    
    return x, u, u_prev, u_next

# --- 2. Numerical Solvers ---

def compute_next_step(u, u_prev, params):
    """
    Calculates the next time step using the Finite Difference Method.
    Returns the new state array u_next.
    """
    c = params['c']
    dt = params['dt']
    dx = params['dx']
    nx = params['nx']
    
    u_next = np.zeros_like(u)
    C2 = (c * dt / dx)**2  # Courant number squared

    # Vectorized calculation (faster than for-loops in Python)
    # u[1:-1] refers to all points from index 1 to the second-to-last
    u_next[1:-1] = 2*u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2*u[1:-1] + u[:-2])
    
    # Boundary Conditions (Fixed ends / Dirichlet: u=0 at ends)
    u_next[0] = 0
    u_next[-1] = 0
    
    return u_next

# --- 3. Visualization & Driver ---

def run_simulation():
    """
    Main driver routine to set up parameters and run the animation.
    """
    # 1. Setup
    params = get_parameters()
    x, u, u_prev, u_next = initialize_fields(params)
    
    # 2. Prepare Plotting
    fig, ax = plt.subplots()
    line, = ax.plot(x, u, color='blue', lw=2)
    
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, params['L'])
    ax.set_title(f"Wave Equation (c={params['c']}, dx={params['dx']:.2f})")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude (u)")
    ax.grid(True)

    # 3. Define Animation Update Wrapper
    # We use a mutable container (list) for state variables so the inner function can update them
    state = [u, u_prev]

    def update(frame):
        current_u, old_u = state
        
        # Compute next step
        new_u = compute_next_step(current_u, old_u, params)
        
        # Shift time steps: old becomes current, current becomes new
        state[1] = current_u
        state[0] = new_u
        
        # Update plot
        line.set_ydata(new_u)
        return line,

    # 4. Execute Animation
    total_frames = int(params['T'] / params['dt'])
    anim = FuncAnimation(fig, update, frames=total_frames, interval=30, blit=True)
    
    #print("Starting simulation window...")
    #plt.show()

    anim.save('wave_animation.gif', writer='pillow', fps=30)
    print("Animation saved as wave_animation.gif")


# --- 4. Entry Point ---

if __name__ == "__main__":
    run_simulation()

