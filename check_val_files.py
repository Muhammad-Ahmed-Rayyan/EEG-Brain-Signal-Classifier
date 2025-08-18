import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "./dataset"

def load_json_data(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_action_samples(dataset_path, action, subject, indices):
    samples = []
    file_paths = []
    for i in indices:
        path = os.path.join(dataset_path, subject, action, f"{subject}_{action}{i}.json")
        if os.path.exists(path):
            samples.append(load_json_data(path))
            file_paths.append(path)
    return np.array(samples), file_paths

# Actions & loading
actions = ["right", "left", "stop", "come"]
#actions = ["danger", "eat", "drink", "help", "restroom"]
raw_data = []
labels = []
paths = []

for action in actions:
    if action == "eat":
        data, fps = load_action_samples(DATASET_PATH, action, "sameed2", range(101, 151))
    else:
        data, fps = load_action_samples(DATASET_PATH, action, "sameed", range(1, 76))
    
    raw_data.extend(data)
    labels.extend([action] * len(data))
    paths.extend(fps)

# Just do the split — no training
_, _, _, _, _, val_paths = train_test_split(
    np.array(raw_data), np.array(labels), np.array(paths),
    test_size=0.2, random_state=42, stratify=labels
)

print("\nValidation files:")
for fp in val_paths:
    print(fp)