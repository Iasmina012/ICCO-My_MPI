import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "measurements.csv")
OUT_PLOT = os.path.join(RESULTS_DIR, "plot.png")

if not os.path.exists(CSV_FILE):
    print("Missing measurements CSV:", CSV_FILE)
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

#ensure correct types for each col
df["procs"] = df["procs"].astype(int)
df["my_mpi_time"] = df["my_mpi_time"].astype(float)
df["mpi_time"] = df["mpi_time"].astype(float)

collectives = list(df["collective"].unique())
collectives.sort()  #alphabetical order

nrows = len(collectives)
ncols = 2

#figure: each row = collective; left col = execution time, right col = speedup
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3 * max(1, nrows)))

if nrows == 1:
    axes = [axes]  #makes it indexable consistently

for i, c in enumerate(collectives):
    sub = df[df["collective"] == c].sort_values("procs")
    procs = sub["procs"].values
    my_times = sub["my_mpi_time"].values
    mpi_times = sub["mpi_time"].values

    #baseline for speedup (measurement with smallest procs)
    if len(procs) == 0:
        continue
    base_procs = procs.min()
    try:
        base_my = sub[sub["procs"] == base_procs]["my_mpi_time"].values[0]
        base_mpi = sub[sub["procs"] == base_procs]["mpi_time"].values[0]
    except Exception:
        base_my = my_times[0]
        base_mpi = mpi_times[0]

    #Execution Time Plot (left side)
    ax_time = axes[i][0] if nrows > 1 else axes[0]
    ax_time.plot(procs, my_times, marker='o', label='My_MPI')
    ax_time.plot(procs, mpi_times, marker='o', label='MPI')
    ax_time.set_xlabel("Processes")
    ax_time.set_ylabel("Time (s)")
    ax_time.set_title(f"{c.capitalize()} — Execution Time")
    ax_time.grid(True)
    ax_time.legend()

    #SpeedUp Plot (right side)
    ax_speed = axes[i][1] if nrows > 1 else axes[1]
    #avoids division by zero
    speed_my = [ (base_my / t) if t > 0 else float('nan') for t in my_times ]
    speed_mpi = [ (base_mpi / t) if t > 0 else float('nan') for t in mpi_times ]
    ax_speed.plot(procs, speed_my, marker='o', label='My_MPI')
    ax_speed.plot(procs, speed_mpi, marker='o', label='MPI')
    ax_speed.set_xlabel("Processes")
    ax_speed.set_ylabel("SpeedUp (T1 / Tno_procs)")
    ax_speed.set_title(f"{c.capitalize()} — SpeedUp")
    ax_speed.grid(True)
    ax_speed.legend()

plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f"Saved combined figure: {OUT_PLOT}")
plt.show()