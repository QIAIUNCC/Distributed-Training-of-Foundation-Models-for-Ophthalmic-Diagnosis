import os

import pandas as pd
import matplotlib.pyplot as plt

from plots.different_loss import read_data

if __name__ == "__main__":
    root = "../centralized/log_100"
    patch4_pattern = os.path.join(root, "*_simmim_ex56_32.txt")
    patch8_pattern = os.path.join(root, "*_simmim_ex44_32.txt")
    patch16_pattern = os.path.join(root, "*_simmim_ex45_32.txt")
    patch32_pattern = os.path.join(root, "*_simmim_ex46_32.txt")

    # Read data from the Hinge and MSE files
    df_4x4 = read_data(patch4_pattern, '4x4')
    df_8x8 = read_data(patch8_pattern, '8x8')
    df_16x16 = read_data(patch16_pattern, '16x16')
    df_32x32 = read_data(patch32_pattern, '32x32')

    df = pd.concat([df_4x4, df_8x8, df_16x16, df_32x32], ignore_index=True)

    # Filter out the data for Center and Random models
    df_4x4 = df[df['Model'] == '4x4']
    df_8x8 = df[df['Model'] == '8x8']
    df_16x16 = df[df['Model'] == '16x16']
    df_32x32 = df[df['Model'] == '32x32']

    # Setting the positions and width for the bars
    pos = list(range(len(df_4x4['Average'])))
    width = 0.15
    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 15
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(pos, df_4x4['Average'], width, alpha=0.7, color='blue', label='4x4')
    plt.bar([p + width for p in pos], df_8x8['Average'], width, alpha=0.7, color='red', label='8x8')
    plt.bar([p + 2*width for p in pos], df_16x16['Average'], width, alpha=0.7, color='green', label='16x16')
    plt.bar([p + 3*width for p in pos], df_32x32['Average'], width, alpha=0.7, color='yellow', label='32x32')

    # Setting the y-axis label, chart title, and ticks
    ax.set_ylabel('Average AUC-ROC')
    # ax.set_title('Comparison of Model Performance (Center vs Random)')
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(df_4x4['Dataset'])

    # Adding the legend and grid
    plt.legend(['4x4', '8x8', '16x16', '32x32'], loc='lower right')
    plt.grid()
    plt.show()
