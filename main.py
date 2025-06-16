import numpy as np
import random
from trail import load_santa_fe_trail
import os

# --- Constants and Hyperparameters ---
# Environment
GRID_SIZE = 32
MAX_STEPS_PER_EPISODE = 250
TOTAL_FOOD_PELLETS_ON_MAP = 89 
START_POS = (0, 0)  # Assuming top-left, adjust based on map
START_ORIENTATION = 'EAST' # Example, adjust based on map (0:E, 1:S, 2:W, 3:N)

# SNN Parameters
FLIF_MEMBRANE_TIME_CONSTANT = 20.0  # ms
FLIF_THRESHOLD_VOLTAGE = 0.7    # mV (or normalized units)
FLIF_RESET_VOLTAGE = 0.0    # mV
FLIF_NEURONS_BIAS = 0.0025    # Small bias
FLIF_FRACTIONAL_ORDER_ALPHA = float(os.environ.get("ALPHA", 0.75))
FLIF_MEMORY_LENGTH = 12500

LIF_MEMBRANE_TIME_CONSTANT = 20.0  # ms
LIF_THRESHOLD_VOLTAGE = 0.75        # mV
LIF_RESET_VOLTAGE = 0.0          # mV
LIF_NEURONS_BIAS = 0.05

DT_NEURON_SIM = 0.1              # ms
T_ANT_DECISION_WINDOW = 5.0       # ms
NUM_NEURON_STEPS_PER_ANT_STEP = int(T_ANT_DECISION_WINDOW / DT_NEURON_SIM)

I_ACTIVE_INPUT_CURRENT = 1.5     # Current injected when context is active


NUM_HIDDEN_NEURONS = 4 # Number of hidden neurons (duh)

# RL Parameters
LEARNING_RATE_ETA = 0.0001
DISCOUNT_FACTOR_GAMMA = 0.99
EXPLORATION_TEMPERATURE_TAU_RL = 1.0 # For softmax

# Surrogate Gradient
SG_RECT_WIDTH = 0.5             # mV

# Action Mapping
ACTION_MAP = {0: 'TurnLeft', 1: 'TurnRight', 2: 'MoveForward'}
ACTION_IDX_MAP = {'TurnLeft':0, 'TurnRight':1, 'MoveForward':2} # For convenience

NUM_EPISODES = 1000 # Example number of training episodes

# --- Helper Functions ---
def calculate_gl_coefficients(alpha, length):
    coeffs = np.zeros(length, dtype=np.float64)
    if length == 0:
        return coeffs
    coeffs[0] = -alpha # Matches user's Cython code
    for j in range(1, length):
        coeffs[j] = (1.0 - (alpha + 1.0) / (j + 1.0)) * coeffs[j-1] # j+1 to match Cython's j indexing
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
    return (np.random.rand(num_weights)) * 0.1 + 0.1

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
        
        # Voltage update based on user's Cython code structure
        # V_new = (-V_old/τm + bias + I_in) * dt^α - GL_history_sum
        # Note: In the Cython code, V_old is on the right with positive sign, effectively making it
        # V[t] = V[t-1] + dt^alpha/tau_m * (-V[t-1] + bias*tau_m + I_in*tau_m) - history*dt^alpha (?)
        # Let's use the one from Cython directly adapted:
        # hidden_layer_voltages[i] = (-hidden_layer_voltages[i] / membrane_time_constant + neurons_bias + cartpole_inputs[i]) * kernel - hidden_history_component[i]
        # This assumes V on LHS is V[t] and V on RHS is V[t-dt] before THIS update.
        
        # Let's re-interpret for a single step. V is current V (V_old for this step)
        # The term (-self.V / self.tau_m + self.bias + input_current) is like dV/dt if alpha=1 and no history
        effective_dV_dt_part = (-self.V / self.tau_m) + self.bias + input_current
        
        self.V = effective_dV_dt_part * kernel - history_component
        # This formulation needs careful check against discrete fractional derivative definitions.
        # A common form is: V[k] = sum_{j=0}^{mem-1} (-1)^j * C(alpha,j) * I[k-j]*h^alpha - (1/tau) * sum_{j=0}^{mem-1} (-1)^j*C(alpha,j)*V[k-j]*h^alpha
        # The user's code seems to be:
        # V_new = (dV_traditional_terms) * dt^alpha - sum(gl_coeffs * V_history)
        # This implies gl_coeffs are for the fractional derivative of V itself.

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
        # Placeholder: Load your Koza trail map here
        # Example: return np.loadtxt(filepath, dtype=int)
        # For now, a simple dummy map for structure
        print(f"Placeholder: Load Koza trail map from {filepath}")
        # Let's create a simple 5x5 map with a small trail for testing structure
        # Replace this with actual Koza map loading
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
        reward = -0.01 # Default step cost
        episode_done_env = False
        food_consumed_flag = False
        prev_pos = list(self.ant_pos)

        if action_str == 'TurnLeft':
            self.ant_orientation_idx = (self.ant_orientation_idx - 1 + 4) % 4
        elif action_str == 'TurnRight':
            self.ant_orientation_idx = (self.ant_orientation_idx + 1) % 4
        elif action_str == 'MoveForward':
            dx, dy = self.orientation_deltas[self.orientations[self.ant_orientation_idx]]
            next_x, next_y = self.ant_pos[0] + dx, self.ant_pos[1] + dy

            if 0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height:
                self.ant_pos = [next_x, next_y]
                if self.trail_map_current[next_y, next_x] == 1:
                    reward = 1.0 # Food reward
                    self.trail_map_current[next_y, next_x] = 0 # Eat food
                    self.food_eaten_in_episode += 1
                    food_consumed_flag = True
                    if self.food_eaten_in_episode == TOTAL_FOOD_PELLETS_ON_MAP: # Use actual total from loaded map
                        episode_done_env = True
            else:
                # Bumped boundary, stays in place, reward is already -0.01 (step_cost)
                pass
        
        return tuple(self.ant_pos), self.orientations[self.ant_orientation_idx], reward, episode_done_env, food_consumed_flag

# --- MAIN SIMULATION AND TRAINING LOOP ---

# Initialize SNN
flif_neuron_params = {
    "alpha": FLIF_FRACTIONAL_ORDER_ALPHA, "tau_m": FLIF_MEMBRANE_TIME_CONSTANT,
    "V_th": FLIF_THRESHOLD_VOLTAGE, "V_reset": FLIF_RESET_VOLTAGE,
    "bias": FLIF_NEURONS_BIAS, "memory_length": FLIF_MEMORY_LENGTH
}


fLIF_food = FractionalLIFNeuron("food_ctx", flif_neuron_params)
fLIF_nofood = FractionalLIFNeuron("nofood_ctx", flif_neuron_params)

''' ### OBSOLETE - Use for non-fractional order ###
lif_leaf_params = {
    "tau_m": LIF_MEMBRANE_TIME_CONSTANT, "V_th": LIF_THRESHOLD_VOLTAGE,
    "V_reset": LIF_RESET_VOLTAGE, "bias_current": LIF_NEURONS_BIAS # Assuming bias is current
}
'''

# Initialize output and hidden neurons
action_leaf_neurons = [FractionalLIFNeuron(f"food_leaf_{i}", flif_neuron_params) for i in range(3)]

hidden_leaf_neurons = [FractionalLIFNeuron(f"hidden_{i}", flif_neuron_params) for i in range(NUM_HIDDEN_NEURONS)]

W_food_to_action = initialize_weights(3)
W_nofood_to_action = initialize_weights(3)

# Initialize Environment
# You need to create "koza_trail.txt" or point to the correct map file.
# For now, using the dummy map in the class. Adjust START_POS and START_ORIENTATION.
environment = SantaFeEnvironment("koza_trail.txt", START_POS, START_ORIENTATION) 
# For dummy map: environment = SantaFeEnvironment("dummy_map.txt", (0,0), 'EAST')


print(f"Starting training for {NUM_EPISODES} episodes...")
avg_food_eaten = 0
food_trace = []


for episode_i in range(NUM_EPISODES):
    ant_pos, ant_orient_str, _ = environment.reset_ant_and_trail()
    
    fLIF_food.reset_state()
    fLIF_nofood.reset_state()
    for leaf_neuron in action_leaf_neurons:
        leaf_neuron.reset_state()

    episode_trajectory = []
    total_episode_reward = 0.0
    food_eaten_this_episode = 0

    for t_ant_step in range(MAX_STEPS_PER_EPISODE):
        is_food_ahead = environment.get_food_ahead()

        active_fLIF = fLIF_food if is_food_ahead else fLIF_nofood
        inactive_fLIF = fLIF_nofood if is_food_ahead else fLIF_food
        
        current_input_to_active_fLIF = I_ACTIVE_INPUT_CURRENT
        current_input_to_inactive_fLIF = 0.0
        
        active_weights = W_food_to_action if is_food_ahead else W_nofood_to_action

        fLIF_spike_trace_this_T_ant = np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP, dtype=int)
        leaf_potentials_this_T_ant = [np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP) for _ in range(3)]
        current_T_ant_leaf_spike_counts = np.zeros(3, dtype=int)

        for t_neuron_idx in range(NUM_NEURON_STEPS_PER_ANT_STEP):
            active_fLIF.update(current_input_to_active_fLIF, DT_NEURON_SIM)
            fLIF_spike_trace_this_T_ant[t_neuron_idx] = active_fLIF.get_spike_state()

            inactive_fLIF.update(current_input_to_inactive_fLIF, DT_NEURON_SIM)
            
            for i_leaf in range(3):
                leaf_neuron = action_leaf_neurons[i_leaf]
                weight = active_weights[i_leaf]
                synaptic_current_to_leaf = fLIF_spike_trace_this_T_ant[t_neuron_idx] * weight
                
                leaf_neuron.update(synaptic_current_to_leaf, DT_NEURON_SIM)
                if leaf_neuron.get_spike_state() == 1:
                    current_T_ant_leaf_spike_counts[i_leaf] += 1
                leaf_potentials_this_T_ant[i_leaf][t_neuron_idx] = leaf_neuron.get_voltage()
        
        action_probabilities = softmax_stable(current_T_ant_leaf_spike_counts / EXPLORATION_TEMPERATURE_TAU_RL)
        
        # Handle case where all spike counts are zero -> uniform probabilities
        if np.sum(current_T_ant_leaf_spike_counts) == 0 :
             action_probabilities = np.ones(3) / 3.0

        chosen_action_idx = np.random.choice(3, p=action_probabilities)
        
        next_ant_pos, next_ant_orient_str, reward, episode_done_env, food_consumed_flag = \
            environment.step(chosen_action_idx)
        
        total_episode_reward += reward
        if food_consumed_flag:
             food_eaten_this_episode +=1

        episode_trajectory.append({
            "is_food_ahead_context": is_food_ahead,
            "fLIF_spike_trace": np.copy(fLIF_spike_trace_this_T_ant),
            "leaf_potentials_traces": [np.copy(p_trace) for p_trace in leaf_potentials_this_T_ant],
            "chosen_action_idx": chosen_action_idx,
            "action_probabilities": np.copy(action_probabilities),
            "reward": reward
        })

        ant_pos, ant_orient_str = next_ant_pos, next_ant_orient_str # Update ant's state string for orientation
        if episode_done_env or food_eaten_this_episode == TOTAL_FOOD_PELLETS_ON_MAP:
            break
    rewards_for_G_t = [t["reward"] for t in episode_trajectory]
    discounted_returns_G_t = calculate_discounted_returns(rewards_for_G_t, DISCOUNT_FACTOR_GAMMA)
    if len(discounted_returns_G_t) > 1:
        mean_G_t = np.mean(discounted_returns_G_t)
        std_G_t = np.std(discounted_returns_G_t)
        if std_G_t > 1e-8: # Add a small epsilon to prevent division by zero if all G_t are the same
            normalized_G_t_values = (discounted_returns_G_t - mean_G_t) / std_G_t
        else:
            normalized_G_t_values = discounted_returns_G_t - mean_G_t # Just center if std is tiny
    elif len(discounted_returns_G_t) == 1:
        normalized_G_t_values = discounted_returns_G_t # Or set to 0 if only one step, as G_0 - mean(G_0) = 0
                                                      # Let's just use it as is for a single step for now
    else: # No transitions
        normalized_G_t_values = np.array([])

    delta_W_food_to_action = np.zeros(3)
    delta_W_nofood_to_action = np.zeros(3)
    for t_idx, transition in enumerate(episode_trajectory):
        G_t = discounted_returns_G_t[t_idx]
        if normalized_G_t_values.size > 0: # Ensure trajectory was not empty
            G_t_to_use = normalized_G_t_values[t_idx] 
        else: # Should not happen if episode_trajectory has items
            G_t_to_use = 0  
        active_weights_grad_delta_ref = delta_W_food_to_action if transition["is_food_ahead_context"] else delta_W_nofood_to_action
        # Determine leaf params based on context (assuming they might differ, though here they are same)
        active_leaf_neuron_params = flif_neuron_params 

        for k_action_idx in range(3): # For each of the 3 active weights
            # Approx dN_k / dw_k 
            # This is where the surrogate gradient for the leaf neuron and presynaptic activity are used
            dNk_dwk_approx = 0.0
            fLIF_spikes_for_grad = transition["fLIF_spike_trace"]
            leaf_k_potentials_for_grad = transition["leaf_potentials_traces"][k_action_idx]
            
            for t_prime_idx in range(len(fLIF_spikes_for_grad)):
                sg_val = rectangular_surrogate_gradient(
                    leaf_k_potentials_for_grad[t_prime_idx],
                    active_leaf_neuron_params["V_th"], # Using V_th from params dict
                    SG_RECT_WIDTH
                )
                if sg_val > 0: # If leaf neuron was "sensitive"
                    dNk_dwk_approx += fLIF_spikes_for_grad[t_prime_idx] # Add presynaptic spike
            
            indicator = 1.0 if k_action_idx == transition["chosen_action_idx"] else 0.0
            prob_ak = transition["action_probabilities"][k_action_idx]
            
            grad_log_pi_component = (indicator - prob_ak) * \
                                    (1.0 / EXPLORATION_TEMPERATURE_TAU_RL) * dNk_dwk_approx
            
            active_weights_grad_delta_ref[k_action_idx] += G_t_to_use * grad_log_pi_component
   
    max_grad_abs_val = 50.0  # **Tune this value carefully!** Start with something like 1.0 or 5.0.

    np.clip(delta_W_food_to_action, -max_grad_abs_val, max_grad_abs_val, out=delta_W_food_to_action)
    np.clip(delta_W_nofood_to_action, -max_grad_abs_val, max_grad_abs_val, out=delta_W_nofood_to_action)

    W_food_to_action += LEARNING_RATE_ETA * delta_W_food_to_action
    W_nofood_to_action += LEARNING_RATE_ETA * delta_W_nofood_to_action
    if (episode_i+1) % 10 == 0: 
        avg_food_eaten += food_eaten_this_episode
        avg_food_eaten /= 10
        print(f"Episode {episode_i+1}: Steps={t_ant_step+1}, Average Food Eaten={avg_food_eaten}, Total Reward={total_episode_reward:.2f}, "
          f"W_food=[{W_food_to_action[0]:.3f}, {W_food_to_action[1]:.3f}, {W_food_to_action[2]:.3f}], "
          f"W_nofood=[{W_nofood_to_action[0]:.3f}, {W_nofood_to_action[1]:.3f}, {W_nofood_to_action[2]:.3f}]")
        avg_food_eaten = 0
    else:
        avg_food_eaten += food_eaten_this_episode
    food_trace.append(food_eaten_this_episode)

print("Training finished. Saving data.")

filename = f"food_eaten_{FLIF_FRACTIONAL_ORDER_ALPHA}.dat"
np.savetxt(filename, food_trace)

print("Data saved.")

# --- TODO ---
# --- 1. Implement _load_map in SantaFeEnvironment with actual Koza trail data.
#    Adjust START_POS and START_ORIENTATION accordingly. Update TOTAL_FOOD_PELLETS_ON_MAP.
# 2. Verify/Refine FractionalLIFNeuron.update to precisely match desired discrete fractional equation.
#    The current adaptation from the Cython code's layer-wise update to a single neuron needs careful checking.
#    Specifically, how V_old and history component interact with kernel:
#    Your Cython: V_new = (Traditional_dV_terms) * kernel - history_component
#    This implies history_component is also scaled by kernel if it's part of a fractional derivative definition like D^alpha(V) = RHS.
#    Or, if D^alpha(V-V_rest) + (V-V_rest)/tau = I, then V updates involve both.
#    The key is to ensure the sum of GL coeffs * V_hist correctly implements the fractional derivative term.
# --- 3. Refine StandardLIFNeuron.update if needed (current is Euler, can use exact integration for LIF).
# 4. Implement more sophisticated dNk_dwk_approx if the simplified one is insufficient (e.g., using eligibility traces).
# 5. ONGOING hyperparameter tuning (learning rates, temperatures, SNN params, fLIF alpha).
