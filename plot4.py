import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def load_and_smooth_data(alpha, avg_window=100):
    filepath = f"food_eaten_{alpha}.dat"
    with open(filepath) as file:
        data = [float(line) for line in file]
    max_val = max(data)
    avg_data = []
    avg = 0
    for i in range(len(data)):
        avg += data[i]
        if (i + 1) % avg_window == 0:
            avg /= avg_window
            avg_data.append(avg)
            avg = 0
    return avg_data, max_val

def load_random_walk_data(filepath="random_walk_avg_food_eaten.dat", max_val=11):
    with open(filepath) as file:
        data = [float(line) for line in file]
    return data, max_val

# Alpha values to sweep
alpha_vals = [0.3, 0.7, 1.0]

# Set up colormap
norm = mcolors.Normalize(vmin=0.0, vmax=1.2)
cmap = cm.inferno

# Storage for histogram data
max_vals = []

# --- First Window: Line Plot ---
plt.figure(figsize=(10, 6))
for alpha in alpha_vals:
    avg_data, max_val = load_and_smooth_data(alpha)
    color = cmap(norm(alpha))
    plt.plot(avg_data, label=f"α={alpha}", color=color)
    max_vals.append((alpha, max_val))

# Add random walk
rw_data, rw_max = load_random_walk_data()
plt.plot(rw_data, label="random walk", color="gray", linestyle="--")
max_vals.append(("random walk", rw_max))

plt.xlabel("Training Episode (x100)")
plt.ylabel("Average Food Eaten")
plt.title("Training Performance by Alpha")
plt.legend()
plt.grid(True)
plt.show()

# --- Second Window: Histogram ---
plt.figure(figsize=(8, 5))
labels, max_values = zip(*sorted(max_vals, key=lambda x: str(x[0])))
colors = [cmap(norm(alpha)) if isinstance(alpha, float) else "gray" for alpha in labels]
plt.bar([str(l) for l in labels], max_values, color=colors)
plt.xlabel("Alpha")
plt.ylabel("Max Food Eaten")
plt.title("Maximum Food Eaten per Alpha")
plt.grid(axis='y')
plt.show()

