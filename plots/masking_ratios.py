import os

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from plots.different_loss import read_data

if __name__ == "__main__":
    root = "../centralized/log_100"
    p10_pattern = os.path.join(root, "*_simmim_ex73_32.txt")
    p20_pattern = os.path.join(root, "*_simmim_ex74_32.txt")
    p30_pattern = os.path.join(root, "*_simmim_ex75_32.txt")
    p40_pattern = os.path.join(root, "*_simmim_ex76_32.txt")
    p50_pattern = os.path.join(root, "*_simmim_ex56_32.txt")
    p60_pattern = os.path.join(root, "*_simmim_ex77_32.txt")
    p70_pattern = os.path.join(root, "*_simmim_ex78_32.txt")
    p80_pattern = os.path.join(root, "*_mim_ex79_32.txt")
    p90_pattern = os.path.join(root, "*_simmim_ex80_32.txt")

    # Read data from the Hinge and MSE files
    df_10 = read_data(p10_pattern, '10%', metric="f1")
    df_20 = read_data(p20_pattern, '20%', metric="f1")
    df_30 = read_data(p30_pattern, '30%', metric="f1")
    df_40 = read_data(p40_pattern, '40%', metric="f1")
    df_50 = read_data(p50_pattern, '50%', metric="f1")
    df_60 = read_data(p60_pattern, '60%', metric="f1")
    df_70 = read_data(p70_pattern, '70%', metric="f1")
    df_80 = read_data(p80_pattern, '80%', metric="f1")
    df_90 = read_data(p90_pattern, '90%', metric="f1")
    df = pd.concat([df_10, df_30, df_40, df_50, df_60, df_70, df_80, df_90], ignore_index=True)
    # Set the font globally
    plt.rcParams['font.family'] = 'Thoma'
    plt.rcParams['font.size'] = 15

    # Plotting
    plt.figure(figsize=(10, 5))

    colors = {
        '10%': 'navy', '20%': 'cyan', '30%': 'red', '40%': 'orange',
        '50%': 'blue', '60%': 'green', '70%': 'purple', '80%': 'brown', '90%': 'pink'
    }

    grouped = df.groupby('Model')
    # Plot each model's average value against the dataset
    for model, group in grouped:
        # If the model is '50%', apply different alpha value
        if model == '50%':
            plt.plot(group['Dataset'], group['Average'], marker='o', label=model, color=colors.get(model, 'black'),
                     alpha=0.5)
        else:
            plt.plot(group['Dataset'], group['Average'], marker='o', label=model, color=colors.get(model, 'black'))

    plt.xlabel('Dataset')
    plt.ylabel('AUC Score(%)')
    # plt.title('Model Performance with Different Across Different Datasets')
    plt.title('')

    plt.legend(title='Masking Percentage')
    plt.grid(True)
    plt.show()
