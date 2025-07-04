from santa_fe_functions import *
# --- Setup for Live Animation ---
import matplotlib.pyplot as plt
plt.ion() # Turn on interactive mode for live plotting

fig, ax = plt.subplots(figsize=(8, 8)) # Create the figure and axes once
plt.show(block=False) # Show the plot window immediately but don't block execution


# Initialize SNN
flif_neuron_params = {
    "alpha": FLIF_FRACTIONAL_ORDER_ALPHA, "tau_m": FLIF_MEMBRANE_TIME_CONSTANT,
    "V_th": FLIF_THRESHOLD_VOLTAGE, "V_reset": FLIF_RESET_VOLTAGE,
    "bias": FLIF_NEURONS_BIAS, "memory_length": FLIF_MEMORY_LENGTH
}

# Context Neurons
fLIF_food = FractionalLIFNeuron("food_ctx", flif_neuron_params)
fLIF_nofood = FractionalLIFNeuron("nofood_ctx", flif_neuron_params)

# Hidden Layer
hidden_neurons = [FractionalLIFNeuron(f"hidden_{i}", flif_neuron_params) for i in range(NUM_HIDDEN_NEURONS)]

# Action Layer (Output)
action_leaf_neurons = [FractionalLIFNeuron(f"action_leaf_{i}", flif_neuron_params) for i in range(3)]

# Weights from Context (2) -> Hidden (N)
W_context_to_hidden = initialize_weights((2, NUM_HIDDEN_NEURONS))

# Weights from Hidden (N) -> Action (3)
W_hidden_to_action = initialize_weights((NUM_HIDDEN_NEURONS, 3)) 

# Initialize Environment
environment = SantaFeEnvironment("koza_trail.txt", START_POS, START_ORIENTATION) 

print(f"Starting training for {FLIF_FRACTIONAL_ORDER_ALPHA} ...")
avg_food_eaten = 0
food_trace = []


for episode_i in range(NUM_EPISODES):
    ant_pos, ant_orient_str, _ = environment.reset_ant_and_trail()
    
    fLIF_food.reset_state()
    fLIF_nofood.reset_state()
    for hn in hidden_neurons:
        hn.reset_state()
    for leaf_neuron in action_leaf_neurons:
        leaf_neuron.reset_state()

    episode_trajectory = []
    total_episode_reward = 0.0
    food_eaten_this_episode = 0

    # --- Initial plot for the new episode ---
    # Clear the previous episode's drawing
    ax.clear() 
    # Plot initial state (ant at (0,0) on the original map)
    plot_ant_trail(
        ax, 
        list(ant_pos), 
        environment.orientations.index(ant_orient_str), 
        np.copy(environment.trail_map_current), # Pass a copy of the initial map
        environment.orientations,
        0, # Step count 0 for initial
        food_eaten_this_episode
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    #plt.pause(0.0) # Pause briefly to show the initial state
    
    for t_ant_step in range(MAX_STEPS_PER_EPISODE):
        is_food_ahead = environment.get_food_ahead()

        active_fLIF = fLIF_food if is_food_ahead else fLIF_nofood
        inactive_fLIF = fLIF_nofood if is_food_ahead else fLIF_food
        
        current_input_to_active_fLIF = I_ACTIVE_INPUT_CURRENT
        current_input_to_inactive_fLIF = 0.0

        fLIF_spike_trace_this_T_ant = np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP, dtype=int)
        hidden_spike_trace_this_T_ant = np.zeros((NUM_NEURON_STEPS_PER_ANT_STEP, NUM_HIDDEN_NEURONS), dtype=int) 
        hidden_potentials_this_T_ant = [np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP) for _ in range(NUM_HIDDEN_NEURONS)] 
        leaf_potentials_this_T_ant = [np.zeros(NUM_NEURON_STEPS_PER_ANT_STEP) for _ in range(3)]
        current_T_ant_leaf_spike_counts = np.zeros(3, dtype=int)


            # --- Simulation Loop ---
        for t_neuron_idx in range(NUM_NEURON_STEPS_PER_ANT_STEP):
            # 1. Update Context Neurons
            active_fLIF.update(current_input_to_active_fLIF, DT_NEURON_SIM)
            inactive_fLIF.update(current_input_to_inactive_fLIF, DT_NEURON_SIM)
            fLIF_spike_trace_this_T_ant[t_neuron_idx] = active_fLIF.get_spike_state()
            
            # Store context spikes
            context_spikes_now = np.zeros(2)
            if is_food_ahead:
                context_spikes_now[0] = active_fLIF.get_spike_state()
            else:
                context_spikes_now[1] = active_fLIF.get_spike_state()

            # 2. Update Hidden Neurons
            # Calculate total current to hidden layer
            synaptic_current_to_hidden = np.dot(context_spikes_now, W_context_to_hidden)
            
            for i_hidden in range(NUM_HIDDEN_NEURONS):
                hidden_neurons[i_hidden].update(synaptic_current_to_hidden[i_hidden], DT_NEURON_SIM)
                spike_state = hidden_neurons[i_hidden].get_spike_state()
                hidden_spike_trace_this_T_ant[t_neuron_idx, i_hidden] = spike_state
                hidden_potentials_this_T_ant[i_hidden][t_neuron_idx] = hidden_neurons[i_hidden].get_voltage()

            # 3. Update Action Neurons
            # Calculate total current to action layer from hidden layer
            hidden_spikes_now = hidden_spike_trace_this_T_ant[t_neuron_idx, :]
            synaptic_current_to_action = np.dot(hidden_spikes_now, W_hidden_to_action)

            for i_leaf in range(3):
                action_leaf_neurons[i_leaf].update(synaptic_current_to_action[i_leaf], DT_NEURON_SIM)
                if action_leaf_neurons[i_leaf].get_spike_state() == 1:
                    current_T_ant_leaf_spike_counts[i_leaf] += 1
                leaf_potentials_this_T_ant[i_leaf][t_neuron_idx] = action_leaf_neurons[i_leaf].get_voltage()

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
            "context_spike_trace": np.copy(fLIF_spike_trace_this_T_ant), # Keeping this for now for simplicity
            "hidden_spike_trace": np.copy(hidden_spike_trace_this_T_ant),
            "hidden_potentials_traces": [np.copy(p_trace) for p_trace in hidden_potentials_this_T_ant],
            "leaf_potentials_traces": [np.copy(p_trace) for p_trace in leaf_potentials_this_T_ant],
            "chosen_action_idx": chosen_action_idx,
            "action_probabilities": np.copy(action_probabilities),
            "reward": reward
        })

        ant_pos, ant_orient_str = next_ant_pos, next_ant_orient_str # Update ant's state string for orientation
        # --- Update the live plot ---
        plot_ant_trail(
            ax, 
            list(ant_pos), # Current ant position
            environment.orientations.index(ant_orient_str), # Current ant orientation index
            np.copy(environment.trail_map_current), # Current state of the map (pass a copy)
            environment.orientations,
            t_ant_step + 1, # Current step count
            food_eaten_this_episode # Food eaten in this episode
        )
        fig.canvas.draw()         # Redraw the canvas
        fig.canvas.flush_events() # Process GUI events
        #plt.pause(0.0)           # Small pause for visual effect (adjust as desired)
 
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
    else: # No transitions
        normalized_G_t_values = np.array([])

    # 1. Initialize gradient matrices for this episode
    delta_W_context_to_hidden = np.zeros_like(W_context_to_hidden)
    delta_W_hidden_to_action = np.zeros_like(W_hidden_to_action)

    # 2. Loop through each step of the episode in reverse (or forward, it's equivalent for this policy gradient form)
    for t_idx, transition in enumerate(episode_trajectory):
        G_t_to_use = normalized_G_t_values[t_idx]
        
        # --- Retrieve data for this transition ---
        chosen_action_idx = transition["chosen_action_idx"]
        action_probabilities = transition["action_probabilities"]
        
        # Reconstruct presynaptic spike activity for both layers
        context_spikes_t = np.zeros(2)
        if transition["is_food_ahead_context"]:
            context_spikes_t[0] = np.sum(transition["context_spike_trace"])
        else:
            context_spikes_t[1] = np.sum(transition["context_spike_trace"])
            
        hidden_spikes_t = np.sum(transition["hidden_spike_trace"], axis=0) # Sum spikes for each hidden neuron over the window

        # --- A. Calculate "error" at the OUTPUT layer ---
        # This is the classic REINFORCE signal: (did_I_take_action - probability_of_action)
        delta_output_layer = np.zeros(3)
        delta_output_layer[chosen_action_idx] = 1.0
        delta_output_layer -= action_probabilities
        
        # Modulate error by the surrogate gradient of the action neurons
        # This approximates how much a change in input would have affected the output spike count
        for k in range(3):
            sg_action = 0.0
            potentials = transition["leaf_potentials_traces"][k]
            for v in potentials:
                sg_action += rectangular_surrogate_gradient(v, flif_neuron_params["V_th"], SG_RECT_WIDTH)
            delta_output_layer[k] *= sg_action

        # --- B. Calculate gradient for W_hidden_to_action ---
        # Formula: ΔW_h_a += pre-synaptic_activity (hidden)  x  post-synaptic_error (output)
        # We use np.outer to multiply the hidden spike vector by the output error vector
        grad_W_hidden_to_action = np.outer(hidden_spikes_t, delta_output_layer)
        delta_W_hidden_to_action += grad_W_hidden_to_action * G_t_to_use

        # --- C. Backpropagate the error to the HIDDEN layer ---
        # The error at the hidden layer is the output error weighted by the connections
        error_at_hidden_layer = np.dot(delta_output_layer, W_hidden_to_action.T)
        
        # Modulate this backpropagated error by the surrogate gradient of the hidden neurons
        for j in range(NUM_HIDDEN_NEURONS):
            sg_hidden = 0.0
            potentials = transition["hidden_potentials_traces"][j]
            for v in potentials:
                sg_hidden += rectangular_surrogate_gradient(v, flif_neuron_params["V_th"], SG_RECT_WIDTH)
            error_at_hidden_layer[j] *= sg_hidden

        # --- D. Calculate gradient for W_context_to_hidden ---
        # Formula: ΔW_c_h += pre-synaptic_activity (context) x post-synaptic_error (hidden)
        grad_W_context_to_hidden = np.outer(context_spikes_t, error_at_hidden_layer)
        delta_W_context_to_hidden += grad_W_context_to_hidden * G_t_to_use

    # 3. Apply the accumulated gradients after the episode
    max_grad_abs_val = 100.0  # I have tried many values here (1.0, 5.0, 10.0, 100.0, etc.)

    np.clip(delta_W_context_to_hidden, -max_grad_abs_val, max_grad_abs_val, out=delta_W_context_to_hidden)
    np.clip(delta_W_hidden_to_action, -max_grad_abs_val, max_grad_abs_val, out=delta_W_hidden_to_action)

    W_context_to_hidden += LEARNING_RATE_ETA * delta_W_context_to_hidden
    W_hidden_to_action += LEARNING_RATE_ETA * delta_W_hidden_to_action


    # Print updates every ten episodes
    avg_food_eaten += food_eaten_this_episode
    if (episode_i+1) % 1 == 0: 
        avg_food_eaten /= 1
        w_c_h_norm = np.linalg.norm(W_context_to_hidden)
        w_h_a_norm = np.linalg.norm(W_hidden_to_action)
        print(f"Episode {episode_i+1}: Steps={t_ant_step+1}, Average Food Eaten={avg_food_eaten}, Total Reward={total_episode_reward:.2f}, "
              f"|W_c_h|={w_c_h_norm:.3f}, |W_h_a|={w_h_a_norm:.3f}")
        avg_food_eaten = 0
    food_trace.append(food_eaten_this_episode)


plt.ioff() # Turn off interactive mode
plt.show() # Keep the final plot window open

print("Training finished. Saving data.")

filename = f"food_eaten_{FLIF_FRACTIONAL_ORDER_ALPHA}.dat"
np.savetxt(filename, food_trace)

print("Data saved.")

