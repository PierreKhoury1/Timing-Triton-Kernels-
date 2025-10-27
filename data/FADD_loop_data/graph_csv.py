import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("sweep_results.csv")

# Filter out any NaNs (for reps = 0)
df = df[df["reps"] > 0]

# --- Plot 1: per-iteration time (ns) from %globaltimer ---
plt.figure(figsize=(8, 5))
plt.plot(df["reps"], df["gt_per_iter_ns"], label="%globaltimer (ns/add)", linewidth=1.5)
plt.xlabel("REPS (number of dependent adds)")
plt.ylabel("Time per iteration (ns)")
plt.title("%globaltimer: per-iteration latency")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: per-iteration cycles from clock64 ---
plt.figure(figsize=(8, 5))
plt.plot(df["reps"], df["clk_per_iter_cycles"], label="clock64 (cycles/add)", color="orange", linewidth=1.5)
plt.xlabel("REPS (number of dependent adds)")
plt.ylabel("Cycles per iteration")
plt.title("clock64: per-iteration latency")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- (optional) overlay both in ns units ---
plt.figure(figsize=(8, 5))
plt.plot(df["reps"], df["gt_per_iter_ns"], label="%globaltimer (ns/add)", linewidth=1.5)
plt.plot(df["reps"], df["clk_per_iter_ns_at_GHz"], label="clock64 converted (ns/add)", color="orange", linewidth=1.5)
plt.xlabel("REPS (number of dependent adds)")
plt.ylabel("Time per iteration (ns)")
plt.title("Comparison: %globaltimer vs clock64")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
