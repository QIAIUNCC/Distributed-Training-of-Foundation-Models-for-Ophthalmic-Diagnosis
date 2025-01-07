import glob
import json
import os.path
import ast
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted


# Function to read data from the files and extract the necessary values
def read_data(file_pattern, model_name, metric="auc"):
    files = natsorted(glob.glob(file_pattern))
    data = {
        'Dataset': [],
        'Model': [],
        'Test-1': [],
        'Test-2': [],
        'Test-3': [],
        'Test-4': [],
        'Test-5': [],
        'Test-6': [],
        'Average': []
    }

    for i, file in enumerate(files):
        if "DS7" in file:
            continue
        data['Dataset'].append(file.split("/")[3][:3].strip())
        data['Model'].append(model_name)
        with open(file, 'r') as f:
            content = f.read().strip().split("==========")
            j = 1
            summation = 0
            for section in content:
                if section.strip():
                    lines = section.splitlines()
                    if len(lines) > 1:
                        string = lines[1].strip().replace('nan', 'null')
                        string = string.replace("'", '"')
                        metrics = json.loads(string)
                        data[f'Test-{j}'].append(metrics[metric])
                        summation += float(metrics[metric])
                        j += 1
        data['Average'].append(summation/6)
                        # data['Average'][int({data["Dataset"][-1][-1])-1]
            # for metric in data:
            #     if
    print(data)

    return pd.DataFrame(data)

if __name__ == "__main__":
    root = "../centralized/log_100"
    hinge_pattern = os.path.join(root, "*_simmim_ex58_32.txt")
    mse_pattern = os.path.join(root, "*_simmim_ex56_32.txt")

    # Read data from the Hinge and MSE files
    df_hinge = read_data(hinge_pattern, 'Hinge')
    df_mse = read_data(mse_pattern, 'MSE')

    # Concatenate the dataframes
    df = pd.concat([df_hinge, df_mse], ignore_index=True)

    # Filter out the data for MSE and Hinge models
    df_mse = df[df['Model'] == 'MSE']
    df_hinge = df[df['Model'] == 'Hinge']

    # Setting the positions and width for the bars
    pos = list(range(len(df_mse['Average'])))
    width = 0.35
    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 15
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(pos, df_mse['Average'], width, alpha=0.7, color='blue', label='MSE')
    plt.bar([p + width for p in pos], df_hinge['Average'], width, alpha=0.7, color='red', label='Hinge')

    # Setting the y-axis label, chart title, and ticks
    ax.set_ylabel('Average AUC-ROC')
    # ax.set_title('Comparison of Model Performance (MSE vs Hinge)')
    ax.set_title('MSE vs. Hinge loss')
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(df_hinge['Dataset'])

    # Adding the legend and grid
    plt.legend(['MSE', 'Hinge'], loc='lower right')
    plt.grid()
    plt.show()
