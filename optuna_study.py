import numpy as np
import random
import os
import optuna # Import Optuna

# Import functions and classes from santa_fe_functions.py
# Assuming santa_fe_functions.py is in the same directory
from santa_fe_functions import (
    SantaFeEnvironment, FractionalLIFNeuron,
    calculate_gl_coefficients, softmax_stable,
    calculate_discounted_returns, rectangular_surrogate_gradient,
    initialize_weights, ACTION_MAP, ACTION_IDX_MAP
)

# --- Fixed Constants (Moved from santa_fe_functions.py for clarity in main) ---
GRID_SIZE = 32
TOTAL_FOOD_PELLETS_ON_MAP = 89
START_POS = (0, 0)
START_ORIENTATION = 'EAST'

# SNN Fixed Parameters (alpha is fixed as per your request)
FLIF_MEMBRANE_TIME_CONSTANT = 20.0  # ms
FLIF_THRESHOLD_VOLTAGE = 0.7    # mV (or normalized units)
FLIF_RESET_VOLTAGE = 0.0    # mV
FLIF_FRACTIONAL_ORDER_ALPHA = 0.75 # FIXED ALPHA VALUE
FLIF_MEMORY_LENGTH = 12500

LIF_MEMBRANE_TIME_CONSTANT = 20.0
LIF_THRESHOLD_VOLTAGE = 0.75
LIF_RESET_VOLTAGE = 0.0

DT_NEURON_SIM = 0.1
T_ANT_DECISION_WINDOW = 5.0
NUM_NEURON_STEPS_PER_ANT_STEP = int(T_ANT_DECISION_WINDOW / DT_NEURON_SIM)

NUM_HIDDEN_NEURONS = 16 # Fixed at 16 as per your last decision

# RL Fixed Parameters
DISCOUNT_FACTOR_GAMMA = 0.99
SG_RECT_WIDTH = 1.2

# Number of Episodes for this Optuna trial
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 50

# --- Optuna Objective Function ---
def objective(trial):
    # --- Tunable Hyperparameters ---
    # LEARNING_RATE_ETA: Log uniform search is typical for learning rates
    LEARNING_RATE_ETA = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3) # A slightly smaller range than previous 0.001
    
    # EXPLORATION_TEMPERATURE_TAU_RL: Log uniform or uniform, depending on expected range
    # Given previous discussion, lower temperature is generally better. Keep range for exploration.
    EXPLORATION_TEMPERATURE_TAU_RL = trial.suggest_uniform('exploration_temperature', 0.1, 1.0) 
    
    # max_grad_abs_val: Uniform or log uniform
    max_grad_abs_val = trial.suggest_uniform('max_grad_abs_val', 1.0, 100.0) # Range based on your previous experiments
    
    # FLIF_NEURONS_BIAS: Affects excitability. Can be uniform.
    FLIF_NEURONS_BIAS = trial.suggest_uniform('neuron_bias', 0.01, 0.1) # Smaller range around your default 0.05
    
    # I_ACTIVE_INPUT_CURRENT: Affects how strongly input neurons fire
    I_ACTIVE_INPUT_CURRENT = trial.suggest_uniform('input_current', 1.0, 3.0) # Range around your default 1.5

    # --- Reward Values (fixed for this study, as per your last decision) ---
    # These are part of the environment, not directly optimized by Optuna in this objective
    # but their values are crucial. They are set within santa_fe_functions.py now.
    # The actual implementation of reward values is in santa_fe_functions.py -> SantaFeEnvironment.step method.
    # For clarity, let's ensure the santa_fe_functions.py you provided has these:
    # reward = -0.005 # General step cost
    # reward = 5.0 # Food reward
    # reward = -5.0 # heavy penalty for bumping walls
    # IMPORTANT: Ensure these are set in santa_fe_functions.py!


    # Initialize SNN with trial-suggested parameters
    flif_neuron_params = {
        "alpha": FLIF_FRACTIONAL_ORDER_ALPHA, "tau_m": FLIF_MEMBRANE_TIME_CONSTANT,
        "V_th": FLIF_THRESHOLD_VOLTAGE, "V_reset": FLIF_RESET_VOLTAGE,
        "bias": FLIF_NEURONS_BIAS, "memory_length": FLIF_MEMORY_LENGTH
    }

    fLIF_food = FractionalLIFNeuron("food_ctx", flif_neuron_params)
    fLIF_nofood = FractionalLIFNeuron("nofood_ctx", flif_neuron_params)
    hidden_neurons = [FractionalLIFNeuron(f"hidden_{i}", flif_neuron_params) for i in range(NUM_HIDDEN_NEURONS)]
    action_leaf_neurons = [FractionalLIFNeuron(f"action_leaf_{i}", flif_neuron_params) for i in range(3)]

    W_context_to_hidden = initialize_weights((2, NUM_HIDDEN_NEURONS))
    W_hidden_to_action = initialize_weights((NUM_HIDDEN_NEURONS, 3))

    environment = SantaFeEnvironment("koza_trail.txt", START_POS, START_ORIENTATION) # Check this path/file exists

    # --- Training Loop ---
    # For Optuna, we'll return the average food eaten over the entire study, or the last few episodes for stability.
    # Let's average over the last 10 episodes for the return value for robustness.
    # food_trace will store food eaten for each episode.
    food_trace_per_episode = []

    for episode_i in range(NUM_EPISODES):
        ant_pos, ant_orient_str, _ = environment.reset_ant_and_trail() #

        fLIF_food.reset_state() #
        fLIF_nofood.reset_state() #
        for hn in hidden_neurons: #
            hn.reset_state() #
        for leaf_neuron in action_leaf_neurons: #
            leaf_neuron.reset_state() #

        episode_trajectory = [] #
        total_episode_reward = 0.0 #
        food_eaten_this_episode = 0 #

        for t_ant_step in range(MAX_STEPS_PER_EPISODE): #
            is_food_ahead = environment.get_food_ahead() #

            active_fLIF = fLIF_food if is_food_ahead else fLIF_nofood #
            inactive_fLIF = fLIF_nofood if is_food_ahead else fLIF_food #
            
            current_input_to_active_fLIF = I_ACTIVE_INPUT_CURRENT #
            current_input_to_inactive_fLIF = 0.0 #

            fLIF_spike_trace_this_T_ant = np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP, dtype=int) #
            hidden_spike_trace_this_T_ant = np.zeros((NUM_NEURON_STEPS_PER_ANT_STEP, NUM_HIDDEN_NEURONS), dtype=int) #
            hidden_potentials_this_T_ant = [np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP) for _ in range(NUM_HIDDEN_NEURONS)] #
            leaf_potentials_this_T_ant = [np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP) for _ in range(3)] #
            current_T_ant_leaf_spike_counts = np.zeros(3, dtype=int) #

            # --- Neuron Simulation Loop ---
            for t_neuron_idx in range(NUM_NEURON_STEPS_PER_ANT_STEP): #
                # 1. Update Context Neurons
                active_fLIF.update(current_input_to_active_fLIF, DT_NEURON_SIM) #
                inactive_fLIF.update(current_input_to_inactive_fLIF, DT_NEURON_SIM) #
                fLIF_spike_trace_this_T_ant[t_neuron_idx] = active_fLIF.get_spike_state() #
                
                context_spikes_now = np.zeros(2) #
                if is_food_ahead: #
                    context_spikes_now[0] = active_fLIF.get_spike_state() #
                else: #
                    context_spikes_now[1] = active_fLIF.get_spike_state() #

                # 2. Update Hidden Neurons
                synaptic_current_to_hidden = np.dot(context_spikes_now, W_context_to_hidden) #
                
                for i_hidden in range(NUM_HIDDEN_NEURONS): #
                    hidden_neurons[i_hidden].update(synaptic_current_to_hidden[i_hidden], DT_NEURON_SIM) #
                    spike_state = hidden_neurons[i_hidden].get_spike_state() #
                    hidden_spike_trace_this_T_ant[t_neuron_idx, i_hidden] = spike_state #
                    hidden_potentials_this_T_ant[i_hidden][t_neuron_idx] = hidden_neurons[i_hidden].get_voltage() #

                # 3. Update Action Neurons
                hidden_spikes_now = hidden_spike_trace_this_T_ant[t_neuron_idx, :] #
                synaptic_current_to_action = np.dot(hidden_spikes_now, W_hidden_to_action) #

                for i_leaf in range(3): #
                    action_leaf_neurons[i_leaf].update(synaptic_current_to_action[i_leaf], DT_NEURON_SIM) #
                    if action_leaf_neurons[i_leaf].get_spike_state() == 1: #
                        current_T_ant_leaf_spike_counts[i_leaf] += 1 #
                    leaf_potentials_this_T_ant[i_leaf][t_neuron_idx] = action_leaf_neurons[i_leaf].get_voltage() #

            action_probabilities = softmax_stable(current_T_ant_leaf_spike_counts / EXPLORATION_TEMPERATURE_TAU_RL) #
            
            if np.sum(current_T_ant_leaf_spike_counts) == 0 : #
                 action_probabilities = np.ones(3) / 3.0 #

            chosen_action_idx = np.random.choice(3, p=action_probabilities) #
            
            next_ant_pos, next_ant_orient_str, reward, episode_done_env, food_consumed_flag = \
                environment.step(chosen_action_idx) #
            
            total_episode_reward += reward #
            if food_consumed_flag: #
                 food_eaten_this_episode +=1 #
            
            episode_trajectory.append({ #
                "is_food_ahead_context": is_food_ahead, #
                "context_spike_trace": np.copy(fLIF_spike_trace_this_T_ant), #
                "hidden_spike_trace": np.copy(hidden_spike_trace_this_T_ant), #
                "hidden_potentials_traces": [np.copy(p_trace) for p_trace in hidden_potentials_this_T_ant], #
                "leaf_potentials_traces": [np.copy(p_trace) for p_trace in leaf_potentials_this_T_ant], #
                "chosen_action_idx": chosen_action_idx, #
                "action_probabilities": np.copy(action_probabilities), #
                "reward": reward #
            })

            ant_pos, ant_orient_str = next_ant_pos, next_ant_orient_str #
            if episode_done_env or food_eaten_this_episode == TOTAL_FOOD_PELLETS_ON_MAP: #
                break #

        rewards_for_G_t = [t["reward"] for t in episode_trajectory] #
        discounted_returns_G_t = calculate_discounted_returns(rewards_for_G_t, DISCOUNT_FACTOR_GAMMA) #
        
        normalized_G_t_values = np.array([]) # Initialize for cases where no transitions or single transition
        if len(discounted_returns_G_t) > 1: #
            mean_G_t = np.mean(discounted_returns_G_t) #
            std_G_t = np.std(discounted_returns_G_t) #
            if std_G_t > 1e-8: #
                normalized_G_t_values = (discounted_returns_G_t - mean_G_t) / std_G_t #
            else: #
                normalized_G_t_values = discounted_returns_G_t - mean_G_t #
        elif len(discounted_returns_G_t) == 1: #
            normalized_G_t_values = discounted_returns_G_t #
        
        delta_W_context_to_hidden = np.zeros_like(W_context_to_hidden) #
        delta_W_hidden_to_action = np.zeros_like(W_hidden_to_action) #

        for t_idx, transition in enumerate(episode_trajectory): #
            G_t_to_use = normalized_G_t_values[t_idx] #
            
            chosen_action_idx = transition["chosen_action_idx"] #
            action_probabilities = transition["action_probabilities"] #
            
            context_spikes_t = np.zeros(2) #
            if transition["is_food_ahead_context"]: #
                context_spikes_t[0] = np.sum(transition["context_spike_trace"]) #
            else: #
                context_spikes_t[1] = np.sum(transition["context_spike_trace"]) #
                
            hidden_spikes_t = np.sum(transition["hidden_spike_trace"], axis=0) #

            delta_output_layer = np.zeros(3) #
            delta_output_layer[chosen_action_idx] = 1.0 #
            delta_output_layer -= action_probabilities #
            
            for k in range(3): #
                sg_action = 0.0 #
                potentials = transition["leaf_potentials_traces"][k] #
                for v in potentials: #
                    sg_action += rectangular_surrogate_gradient(v, flif_neuron_params["V_th"], SG_RECT_WIDTH) #
                delta_output_layer[k] *= sg_action #

            grad_W_hidden_to_action = np.outer(hidden_spikes_t, delta_output_layer) #
            delta_W_hidden_to_action += grad_W_hidden_to_action * G_t_to_use #

            error_at_hidden_layer = np.dot(delta_output_layer, W_hidden_to_action.T) #
            
            for j in range(NUM_HIDDEN_NEURONS): #
                sg_hidden = 0.0 #
                potentials = transition["hidden_potentials_traces"][j] #
                for v in potentials: #
                    sg_hidden += rectangular_surrogate_gradient(v, flif_neuron_params["V_th"], SG_RECT_WIDTH) #
                error_at_hidden_layer[j] *= sg_hidden #

            grad_W_context_to_hidden = np.outer(context_spikes_t, error_at_hidden_layer) #
            delta_W_context_to_hidden += grad_W_context_to_hidden * G_t_to_use #

        # Apply gradient clipping and updates
        np.clip(delta_W_context_to_hidden, -max_grad_abs_val, max_grad_abs_val, out=delta_W_context_to_hidden) #
        np.clip(delta_W_hidden_to_action, -max_grad_abs_val, max_grad_abs_val, out=delta_W_hidden_to_action) #

        W_context_to_hidden += LEARNING_RATE_ETA * delta_W_context_to_hidden #
        W_hidden_to_action += LEARNING_RATE_ETA * delta_W_hidden_to_action #
        
        food_trace_per_episode.append(food_eaten_this_episode) #

        # Optuna pruning: Report intermediate value every 10 episodes
        if (episode_i + 1) % 10 == 0:
            avg_food_last_10 = np.mean(food_trace_per_episode[-10:])
            print(f"Episode {episode_i+1}: Average Food Eaten={avg_food_last_10}", flush=True)
            trial.report(avg_food_last_10, episode_i)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Return average food eaten from the last portion of the training for the objective
    # This helps smooth out noisy final performance.
    return np.mean(food_trace_per_episode[-50:]) # Average over last 50 episodes for robustness

if __name__ == '__main__':
    # --- Create and Run Optuna Study ---
    study = optuna.create_study(direction='maximize') # We want to maximize food eaten
    study.optimize(objective, n_trials=50) # Run 50 trials (you can adjust this number)

    print("\n--- Optuna Study Results ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Save study results
    # import joblib
    # joblib.dump(study, "santa_fe_study.pkl")
