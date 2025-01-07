import os

import pandas as pd
import matplotlib.pyplot as plt

from plots.different_loss import read_data


if __name__ == "__main__":
    root = "../centralized/log_100"
    random_pattern = os.path.join(root, "*_simmim_ex41_32.txt")
    center_pattern = os.path.join(root, "*_simmim_ex56_32.txt")

    # Read data from the Hinge and MSE files
    df_random = read_data(random_pattern, 'Random')
    df_center = read_data(center_pattern, 'Center')

    df = pd.concat([df_random, df_center], ignore_index=True)


    # Filter out the data for Center and Random models
    df_center = df[df['Model'] == 'Center']
    df_random = df[df['Model'] == 'Random']

    # Setting the positions and width for the bars
    pos = list(range(len(df_center['Average'])))
    width = 0.35
    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 15
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(pos, df_center['Average'], width, alpha=0.7, color='blue', label='Center')
    plt.bar([p + width for p in pos], df_random['Average'], width, alpha=0.7, color='red', label='Random')

    # Setting the y-axis label, chart title, and ticks
    ax.set_ylabel('Average AUC-ROC')
    # ax.set_title('Comparison of Model Performance (Center vs Random)')
    ax.set_title('Center vs. Random masking')
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(df_random['Dataset'])

    # Adding the legend and grid
    plt.legend(['Center', 'Random'], loc='lower right')
    plt.grid()
    plt.show()
