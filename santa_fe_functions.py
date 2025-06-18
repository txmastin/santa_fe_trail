import numpy as np
import random
from trail import load_santa_fe_trail
import os

# --- Constants and Hyperparameters ---
# Environment
GRID_SIZE = 32
MAX_STEPS_PER_EPISODE = 100
TOTAL_FOOD_PELLETS_ON_MAP = 89
START_POS = (0, 0)  # top-left
START_ORIENTATION = 'EAST' # Example, adjust based on map (0:E, 1:S, 2:W, 3:N)

# SNN Parameters
FLIF_MEMBRANE_TIME_CONSTANT = 20.0  # ms
FLIF_THRESHOLD_VOLTAGE = 0.7    # mV (or normalized units)
FLIF_RESET_VOLTAGE = 0.0    # mV
FLIF_NEURONS_BIAS = 0.1917    # Small bias
FLIF_FRACTIONAL_ORDER_ALPHA = float(os.environ.get("ALPHA", 0.5))
FLIF_MEMORY_LENGTH = 12500 # full context for 250 total steps (change if necessary for longer simulations)

LIF_MEMBRANE_TIME_CONSTANT = 20.0  # ms
LIF_THRESHOLD_VOLTAGE = 0.75        # mV
LIF_RESET_VOLTAGE = 0.0          # mV
LIF_NEURONS_BIAS = 0.05

DT_NEURON_SIM = 0.1              # ms
T_ANT_DECISION_WINDOW = 5.0       # ms
NUM_NEURON_STEPS_PER_ANT_STEP = int(T_ANT_DECISION_WINDOW / DT_NEURON_SIM)

I_ACTIVE_INPUT_CURRENT = 0.5     # Current injected when context is active

NUM_HIDDEN_NEURONS = 4 # Number of hidden neurons (duh)

# RL Parameters
LEARNING_RATE_ETA = 0.0001
DISCOUNT_FACTOR_GAMMA = 0.99
EXPLORATION_TEMPERATURE_TAU_RL = 1.0

# Surrogate Gradient
SG_RECT_WIDTH = 1.2             # mV

# Action Mapping
ACTION_MAP = {0: 'TurnLeft', 1: 'TurnRight', 2: 'MoveForward'}
ACTION_IDX_MAP = {'TurnLeft':0, 'TurnRight':1, 'MoveForward':2} # For convenience

NUM_EPISODES = 1000 # Example number of training episodes


# --- Helper Functions ---
def calculate_gl_coefficients(alpha, length):
    coeffs = np.zeros(length, dtype=np.float64)
    if length == 0:
        return coeffs
    coeffs[0] = -alpha
    for j in range(1, length):
        coeffs[j] = (1.0 - (alpha + 1.0) / (j + 1.0)) * coeffs[j-1]
    return coeffs

def softmax_stable(logits_array):
    if not logits_array.size: return np.array([]) # Handle empty array
    stable_logits = logits_array - np.max(logits_array)
    exp_logits = np.exp(stable_logits)
    sum_exp_logits = np.sum(exp_logits)
    if sum_exp_logits == 0: # Avoid division by zero if all logits are extremely small
        return np.ones_like(exp_logits) / exp_logits.size
    return exp_logits / sum_exp_logits

def calculate_discounted_returns(rewards_list, gamma):
    G = 0.0
    discounted_returns = np.zeros_like(rewards_list, dtype=float)
    for t in reversed(range(len(rewards_list))):
        G = rewards_list[t] + gamma * G
        discounted_returns[t] = G
    return discounted_returns

def rectangular_surrogate_gradient(membrane_potential, threshold, width):
    u = membrane_potential - threshold
    if abs(u) < width / 2.0:
        return 1.0 / width
    return 0.0

def initialize_weights(num_weights):
    return (np.random.rand(*num_weights)) * 0.1



# --- Neuron Classes ---
class FractionalLIFNeuron:
    def __init__(self, neuron_id, params):
        self.neuron_id = neuron_id
        self.alpha = params.get("alpha", FLIF_FRACTIONAL_ORDER_ALPHA)
        self.tau_m = params.get("tau_m", FLIF_MEMBRANE_TIME_CONSTANT)
        self.V_th = params.get("V_th", FLIF_THRESHOLD_VOLTAGE)
        self.V_reset = params.get("V_reset", FLIF_RESET_VOLTAGE)
        self.bias = params.get("bias", FLIF_NEURONS_BIAS)
        self.memory_length = params.get("memory_length", FLIF_MEMORY_LENGTH)

        self.V = self.V_reset
        self.voltage_history = np.full(self.memory_length, self.V_reset, dtype=np.float64)
        self.gl_coefficients = calculate_gl_coefficients(self.alpha, self.memory_length)
        self.spike_state = 0

    def reset_state(self):
        self.V = self.V_reset
        self.voltage_history.fill(self.V_reset)
        self.spike_state = 0

    def update(self, input_current, dt):
        self.spike_state = 0

        history_component = 0.0
        if self.memory_length > 0:
            history_component = np.dot(self.gl_coefficients, self.voltage_history)

        kernel = dt**self.alpha

        effective_dV_dt_part = (-self.V / self.tau_m) + self.bias + input_current

        self.V = effective_dV_dt_part * kernel - history_component

        if self.V >= self.V_th:
            self.spike_state = 1
            self.V = self.V_reset

        # Update voltage_history (roll and add new V)
        if self.memory_length > 0:
            self.voltage_history = np.roll(self.voltage_history, 1)
            self.voltage_history[0] = self.V # Store post-reset or current subthreshold V

    def get_spike_state(self):
        return self.spike_state

    def get_voltage(self):
        return self.V


### Non-fractional neuron model ###
class StandardLIFNeuron:
    def __init__(self, neuron_id, params):
        self.neuron_id = neuron_id
        self.tau_m = params.get("tau_m", LIF_MEMBRANE_TIME_CONSTANT)
        self.V_th = params.get("V_th", LIF_THRESHOLD_VOLTAGE)
        self.V_reset = params.get("V_reset", LIF_RESET_VOLTAGE)
        self.bias_current = params.get("bias_current", LIF_NEURONS_BIAS) # Assuming bias is a current

        self.V = self.V_reset
        self.spike_state = 0

    def reset_state(self):
        self.V = self.V_reset
        self.spike_state = 0

    def update(self, input_current, dt):
        self.spike_state = 0
        # dV/dt = (-V + V_rest + bias_current*R_m + input_current*R_m) / tau_m
        # Assuming V_rest = 0 and R_m is absorbed into currents or tau_m definition.
        # More simply: dV/dt = (-V/tau_m) + (bias_current + input_current)/C_m
        # If tau_m = R_m * C_m, then dV/dt = (-V + R_m*(bias_current + input_current))/tau_m
        # Let's assume bias is a current and input_current is also a current.
        # dV = ((-self.V / self.tau_m) + self.bias_current + input_current) * dt
        # A common discrete form:
        alpha_decay = np.exp(-dt / self.tau_m)
        self.V = self.V * alpha_decay + (self.bias_current + input_current) * (1 - alpha_decay) * self.tau_m # if tau_m is R*C and I is current
        # Simpler Euler:
        # dV_dt = (-self.V + self.bias_current*self.tau_m + input_current*self.tau_m) / self.tau_m # if bias is a voltage-like term
        dV_dt = (-self.V / self.tau_m) + self.bias_current + input_current # if bias and input_current are currents scaled by 1/C
        self.V += dV_dt * dt


        if self.V >= self.V_th:
            self.spike_state = 1
            self.V = self.V_reset

    def get_spike_state(self):
        return self.spike_state

    def get_voltage(self):
        return self.V

# --- Environment Class ---
class SantaFeEnvironment:
    def __init__(self, map_filepath, start_pos, start_orientation_str):
        self.trail_map_original = self._load_map(map_filepath)
        self.trail_map_current = np.copy(self.trail_map_original)
        self.start_pos = start_pos
        self.start_orientation_str = start_orientation_str # 'EAST', 'SOUTH', 'WEST', 'NORTH'
        self.orientations = ['EAST', 'SOUTH', 'WEST', 'NORTH']
        self.orientation_deltas = { # dx, dy
            'EAST': (1, 0), 'SOUTH': (0, 1), 'WEST': (-1, 0), 'NORTH': (0, -1)
        }
        self.ant_pos = None
        self.ant_orientation_idx = None # Index in self.orientations
        self.food_eaten_in_episode = 0
        self.grid_height, self.grid_width = self.trail_map_original.shape

    def _load_map(self, filepath):
        simple_map = np.zeros((5,5), dtype=int)
        simple_map[0,1] = 1; simple_map[0,2] = 1; simple_map[0,3] = 1;
        return load_santa_fe_trail()

    def reset_ant_and_trail(self):
        self.trail_map_current = np.copy(self.trail_map_original)
        self.ant_pos = list(self.start_pos)
        self.ant_orientation_idx = self.orientations.index(self.start_orientation_str)
        self.food_eaten_in_episode = 0
        return tuple(self.ant_pos), self.orientations[self.ant_orientation_idx], np.sum(self.trail_map_current)

    def get_food_ahead(self):
        dx, dy = self.orientation_deltas[self.orientations[self.ant_orientation_idx]]
        front_x, front_y = self.ant_pos[0] + dx, self.ant_pos[1] + dy

        if 0 <= front_x < self.grid_width and 0 <= front_y < self.grid_height:
            return self.trail_map_current[front_y, front_x] == 1 # Assuming map is (y,x)
        return False # Off grid means no food

    def step(self, action_idx): # action_idx: 0:L, 1:R, 2:Fwd
        action_str = ACTION_MAP[action_idx]
        reward = -0.005 # Default step cost
        episode_done_env = False
        food_consumed_flag = False
        prev_pos = list(self.ant_pos)

        if action_str == 'TurnLeft':
            self.ant_orientation_idx = (self.ant_orientation_idx - 1 + 4) % 4
            dx, dy = self.orientation_deltas[self.orientations[self.ant_orientation_idx]]
            next_x, next_y = self.ant_pos[0] + dx, self.ant_pos[1] + dy
        elif action_str == 'TurnRight':
            self.ant_orientation_idx = (self.ant_orientation_idx + 1) % 4
            dx, dy = self.orientation_deltas[self.orientations[self.ant_orientation_idx]]
            next_x, next_y = self.ant_pos[0] + dx, self.ant_pos[1] + dy
        elif action_str == 'MoveForward':
            dx, dy = self.orientation_deltas[self.orientations[self.ant_orientation_idx]]
            next_x, next_y = self.ant_pos[0] + dx, self.ant_pos[1] + dy

        if 0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height:
            self.ant_pos = [next_x, next_y]
            if self.trail_map_current[next_y, next_x] == 1:
                reward = 2.0 # Food reward
                self.trail_map_current[next_y, next_x] = 0 # Eat food
                self.food_eaten_in_episode += 1
                food_consumed_flag = True
                if self.food_eaten_in_episode == TOTAL_FOOD_PELLETS_ON_MAP: # Use actual total from loaded map
                    episode_done_env = True
        else:
            reward = -0.1 # penalty for bumping walls
            pass

        return tuple(self.ant_pos), self.orientations[self.ant_orientation_idx], reward, episode_done_env, food_consumed_flag

# --- Visualization Functions ---
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_ant_trail(ax, ant_pos, ant_orientation_idx, trail_map, orientations, step_count, food_eaten_current_episode):
    ax.clear()
    
    # Plot the trail (food pellets)
    # Using 'YlOrRd' for food, 'binary' for trail seems fine. Adjust cmap as desired.
    ax.imshow(trail_map, cmap='YlOrRd', origin='upper', extent=[0, trail_map.shape[1], trail_map.shape[0], 0])
    
    # Draw the ant
    ant_x, ant_y = ant_pos[0], ant_pos[1]
    
    # Ant shape (simple triangle for orientation, centered in cell)
    # Adjust coordinates to center the ant in its grid cell (0.5 offset)
    center_x, center_y = ant_x + 0.5, ant_y + 0.5
    
    # Base triangle points (relative to center)
    base_size = 0.3 # Size of the triangle from center to tip
    
    if orientations[ant_orientation_idx] == 'EAST':
        triangle = plt.Polygon([
            [center_x - base_size, center_y - base_size], 
            [center_x + base_size, center_y], 
            [center_x - base_size, center_y + base_size]
        ], color='blue') # Changed ant color to blue for better contrast
    elif orientations[ant_orientation_idx] == 'SOUTH':
        triangle = plt.Polygon([
            [center_x - base_size, center_y - base_size], 
            [center_x, center_y + base_size], 
            [center_x + base_size, center_y - base_size]
        ], color='blue')
    elif orientations[ant_orientation_idx] == 'WEST':
        triangle = plt.Polygon([
            [center_x + base_size, center_y - base_size], 
            [center_x - base_size, center_y], 
            [center_x + base_size, center_y + base_size]
        ], color='blue')
    elif orientations[ant_orientation_idx] == 'NORTH':
        triangle = plt.Polygon([
            [center_x - base_size, center_y + base_size], 
            [center_x, center_y - base_size], 
            [center_x + base_size, center_y + base_size]
        ], color='blue')
            
    ax.add_patch(triangle)
    
    ax.set_xticks(np.arange(0, trail_map.shape[1], 1))
    ax.set_yticks(np.arange(0, trail_map.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_title(f"Step: {step_count}, Food: {food_eaten_current_episode}")
    ax.set_xlim(-0.5, trail_map.shape[1] - 0.5)
    ax.set_ylim(trail_map.shape[0] - 0.5, -0.5) # Invert y-axis for top-left (0,0)

def animate_episode(episode_states, env_orientations, episode_number, filename=None):
    if not episode_states:
        print(f"No states to animate for Episode {episode_number}.")
        return

    fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted for better visibility
    fig.suptitle(f"Episode {episode_number}") # Title for the whole figure

    # The update function for the animation
    def update(frame):
        state = episode_states[frame]
        plot_ant_trail(
            ax, 
            state['pos'], 
            state['orientation_idx'], 
            state['trail_map'], 
            env_orientations,
            state['step_count'],
            state['food_eaten_current_episode']
        )
        return ax, # Return iterable for blitting (optional but good practice)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(episode_states), 
        interval=50, # Milliseconds between frames (adjust for desired speed)
        blit=False, # Set to True for performance, but can cause issues with some plots
        repeat=False # Play once
    )
    
    # Close the plot immediately after creating the animation object
    # This prevents many static plots from appearing
    plt.close(fig) 

    if filename:
        print(f"Saving animation for Episode {episode_number} to {filename}...")
        try:
            # Make sure you have 'imagemagick' installed and in your system's PATH
            # For MP4, you'd use writer='ffmpeg' and filename.mp4
            ani.save(filename, writer='imagemagick', fps=20) 
            print(f"Animation for Episode {episode_number} saved!")
        except Exception as e:
            print(f"Error saving animation for Episode {episode_number}: {e}")
            print("Make sure 'imagemagick' (for GIF) or 'ffmpeg' (for MP4) is installed and accessible in your system's PATH.")
    else:
        # If running in a Jupyter Notebook, you can display it directly
        from IPython.display import HTML
        return HTML(ani.to_jshtml())
