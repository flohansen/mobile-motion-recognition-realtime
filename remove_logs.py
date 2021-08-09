import os
import shutil

experiment_dir = 'experiments'
logs_dir = 'logs'

experiments = os.listdir(experiment_dir)
logs_to_delete = [log for log in os.listdir(logs_dir) if log not in experiments]

for log in logs_to_delete:
    shutil.rmtree(os.path.join(logs_dir, log))