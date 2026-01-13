import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.results = []
        os.makedirs(log_dir, exist_ok=True)

    def log_task(self, task_id, accuracies):
        self.results.append({"training_task_id": task_id, "accuracies": accuracies})

    def save(self, config_dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)

        full_log = {"config": config_dict, "results": self.results}

        with open(filepath, "w") as f:
            json.dump(full_log, f, indent=4)
        print(f"✅ Results saved to: {filepath}")
