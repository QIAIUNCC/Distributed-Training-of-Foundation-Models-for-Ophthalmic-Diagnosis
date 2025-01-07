import os
import json
import re

import numpy as np
from natsort import natsorted

import glob
def is_null_like(value):
    null_like = [np.nan, 'nan', None, 'Null', 'null']
    return value in null_like or (isinstance(value, float) and np.isnan(value))


def read_val_results(file_path):
    if not os.path.exists(file_path):
        return False

    # Open the file and load the JSON content into a Python dictionary
    with open(file_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict[0]

def write_val_results(file_path, result):
    # Open the file and load the JSON content into a Python dictionary
    with open(file_path, "w") as json_file:
        json.dump(result, json_file, indent=4) 

def log_results(classes, results, client_name, comment, config, log_name=None, approach="FL"):
    result = {}
    metrics = ["accuracy", "precision", "auc", "f1"]
    result["auc"] = results[0]["test_auc"]
    result["pr"] = results[0]["test_pr"]
    result["f1"] = results[0]["test_f1"]
    result["accuracy"] = results[0]["test_accuracy"]
    result["precision"] = results[0]["test_precision"]
    result["loss"] = results[0]["test_loss"]
    for c in classes.keys():
        for m in metrics:
            result[f"test_{m}_" + c] = results[0][f"test_{m}_" + c]
    result = {key: round(value, 4) for key, value in result.items()}
    dir_name = f"log_{config['epochs']}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    arch = comment.replace(')', '').replace('(', '').replace(',', '').replace("'", "")
    if log_name is None:
        log_name = f'{dir_name}/{client_name}_{approach}_{arch}_{config["batch_size"]}.txt'
    if os.path.exists(log_name) or approach != "fl":
        file_mode = "a+"
    else:
        file_mode = "w"
    separator_line = f"============={config['current_round']}=======================\n"
    with open(log_name, file_mode) as f:
        if file_mode == "w":
            f.write(separator_line)  # Write the line if not found
        elif approach == "fl" and file_mode != "w":
            f.seek(0)  # Go to the beginning of the file to read
            content = f.read()  # Read the entire content of the file
            # Check if the specific line is already in the file
            if separator_line not in content:
                f.write(separator_line)  # Write the line if not found
        f.write(f"=========={config['test_set']}==========")
        f.write('\n')
        f.write(str(result))
        f.write('\n')
            
def extract_f1_scores(content, ds_name):
    """
    Extracts F1 scores for all DS entries and returns the F1 score for the specified DS
    and the sum of F1 scores for the others.
    """
    f1_scores = {}

    # Regex to find dataset entries and their F1 scores
    matches = re.findall(r"==========DS(\d+)==========\n{([^}]+)}", content)
    for match in matches:
        ds_num, ds_content = match
        f1_match = re.search(r"'f1': (\d+\.\d+)", ds_content)
        if f1_match:
            f1_scores[f"DS{ds_num}"] = float(f1_match.group(1))

    specific_f1 = f1_scores.get(ds_name, 0)  # F1 score for the dataset corresponding to the file name
    other_f1_sum = sum(f1 for ds, f1 in f1_scores.items())  # Sum of F1 scores for the other datasets

    return specific_f1, other_f1_sum

if __name__ == "__main__":
    # Define the directory containing the files
    directory = ""  # Replace with your directory path
    search_pattern = "local_8"
    # Initialize a list to store the AUC values across all files
    all_auc_values = []
    print(glob.glob(os.path.join(directory, f"*{search_pattern}*")))
    # Iterate over all files in the directory
    for file_path in natsorted(glob.glob(os.path.join(directory, f"*{search_pattern}*"))):
        if os.path.isfile(file_path):  # Ensure it's a file, not a subdirectory
            with open(file_path, "r") as file:
                for line in file:
                    if line.startswith("{'auc':"):
                        # Extract the AUC value from the line
                        auc_value = float(line.split("'auc': ")[1].split(",")[0])
                        all_auc_values.append(auc_value)

        # Convert AUC values to percentages and calculate the average
        auc_percentages = [round(value * 100, 2) for value in all_auc_values]
        average_auc = round(sum(auc_percentages) / len(auc_percentages), 2) if auc_percentages else 0

        # Print the output in the desired format
        output = " & ".join(f"{value}" for value in auc_percentages) + f" & {average_auc}"
        print(f"============================={file_path}")
        print(output)
        all_auc_values.clear()
        
