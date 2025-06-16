import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def load_and_smooth_data(alpha, avg_window=100):
    filepath = f"food_eaten_{alpha}.dat"
    with open(filepath) as file:
        data = [float(line) for line in file]
    
    max_val = max(data)

    # Compute mean and std per window
    data = np.array(data)
    num_windows = len(data) // avg_window
    data = data[:num_windows * avg_window].reshape((num_windows, avg_window))
    means = data.mean(axis=1)
    stds = data.std(axis=1)

    return means, stds, max_val

# Alpha values to sweep
alpha_vals = [0.3, 0.7, 1.0]

# Set up colormap
norm = mcolors.Normalize(vmin=min(alpha_vals), vmax=max(alpha_vals))
cmap = cm.viridis

# Storage for histogram data
max_vals = []

# --- First Window: Line Plot with Std Deviation Bands ---
plt.figure(figsize=(10, 6))
for alpha in alpha_vals:
    means, stds, max_val = load_and_smooth_data(alpha)
    color = cmap(norm(alpha))
    x = np.arange(len(means))
    
    # Plot mean line
    plt.plot(x, means, label=f"α={alpha}", color=color)
    
    # Fill ±1 std dev
    plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.3)
    
    max_vals.append((alpha, max_val))

plt.xlabel("Training Episode (x100)")
plt.ylabel("Average Food Eaten")
plt.title("Training Performance by Alpha (Mean ± Std Dev)")
plt.legend()
plt.grid(True)
plt.show()

# --- Second Window: Histogram of Max Values ---
plt.figure(figsize=(8, 5))
alphas_sorted, max_values_sorted = zip(*sorted(max_vals))
colors = [cmap(norm(alpha)) for alpha in alphas_sorted]
plt.bar([str(a) for a in alphas_sorted], max_values_sorted, color=colors)
plt.xlabel("Alpha")
plt.ylabel("Max Food Eaten")
plt.title("Maximum Food Eaten per Alpha")
plt.grid(axis='y')
plt.show()

