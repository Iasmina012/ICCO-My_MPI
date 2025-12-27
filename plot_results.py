import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "measurements.csv")
GRAPH_FILE = os.path.join(RESULTS_DIR, "plot.png")

if not os.path.exists(CSV_FILE):
    print("Missing measurements CSV:", CSV_FILE)
    sys.exit(1)

data = pd.read_csv(CSV_FILE)

#ensure numeric type
data["procs"] = data["procs"].astype(int)
data["my_mpi_time"] = data["my_mpi_time"].astype(float)
data["mpi_time"] = data["mpi_time"].astype(float)

#calculates speedups relative to 1 proc
T1_my_mpi = data.loc[data["procs"] == 1, "my_mpi_time"].values[0]
T1_mpi = data.loc[data["procs"] == 1, "mpi_time"].values[0]
data["speedup_my_mpi"] = T1_my_mpi / data["my_mpi_time"]
data["speedup_mpi"] = T1_mpi / data["mpi_time"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.plot(data["procs"], data["my_mpi_time"], marker="o", label="My_MPI")
ax1.plot(data["procs"], data["mpi_time"], marker="o", label="MPI")
ax1.set_xlabel("Number of Processes")
ax1.set_ylabel("Execution Time (s)")
ax1.set_title("Execution Time")
ax1.grid(True)
ax1.legend()

ax2.plot(data["procs"], data["speedup_my_mpi"], marker="o", label="My_MPI")
ax2.plot(data["procs"], data["speedup_mpi"], marker="o", label="MPI")
ax2.set_xlabel("Number of Processes")
ax2.set_ylabel("SpeedUp (T1 / Tno_procs)")
ax2.set_title("SpeedUp")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(GRAPH_FILE)
print(f"Plot saved as {GRAPH_FILE}")
plt.show()