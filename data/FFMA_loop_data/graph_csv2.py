import pandas as pd
import matplotlib.pyplot as plt

GHz = 1.680  # same as used in your experiment
ns_per_cycle = 1.0 / GHz

df = pd.read_csv("sweep_results.csv")
df = df[df["reps"] > 0]

# Convert cycles → nanoseconds for clock64
df["clk_mean_ns"] = df["clk_mean_cycles"] * ns_per_cycle

# --- Plot: total measured timing vs REPS ---
plt.figure(figsize=(8, 5))
plt.plot(df["reps"], df["gt_mean_ns"], label="%globaltimer (total Δt, ns)", linewidth=1.5)
plt.plot(df["reps"], df["clk_mean_ns"], label="clock64 (total Δt, ns converted)", linewidth=1.5, color="orange")
plt.xlabel("REPS (number of dependent adds)")
plt.ylabel("Measured duration (ns)")
plt.title("Total timing region vs number of repeated instructions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
