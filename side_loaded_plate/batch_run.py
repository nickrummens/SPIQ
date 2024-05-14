import sys
import subprocess

net_types = ["pfnn", "spinn"]
logged_fields = ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy']
available_time = 3*60
n_iter = int(1e8)
num_runs = 1
log_every = 10000

for net_type in net_types:
    for run in range(num_runs):
        try:
            print(f"Running with net_type={net_type}, run={run}")
            subprocess.check_call([sys.executable, "elasticity_plate.py", f"--net_type={net_type}", f"--log_every={log_every}", f"--available_time={available_time}"])
        except subprocess.CalledProcessError as e:
            print(f"Run with net_type={net_type}, run={run} failed")
            print(e)
            sys.exit(1)