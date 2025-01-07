import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# Example data
data = {
    'DS1': [80.01, 79.85, 78.76, 80.87],
    'DS2': [70.25, 66.67, 63.46, 61.75],
    'DS3': [68.86, 67.31, 63.96, 62.87],
    'DS4': [83.21, 80.24, 75.31, 78.95],
    'DS5': [58.30, 58.52, 55.84, 60.58],
    'DS6': [82.66, 76.75, 76.77, 73.65]
}
models = ['30', '40', '50%', '60%', '70%', '90%']
df = pd.DataFrame(data, index=models)

# Set the font globally
mpl.rcParams['font.family'] = 'Times New Roman'

# Plotting
plt.figure(figsize=(10, 5))
plt.title('Example Plot', fontname='Times New Roman', fontsize=14)
plt.xlabel('X Axis', fontname='Times New Roman', fontsize=12)
plt.ylabel('Y Axis', fontname='Times New Roman', fontsize=12)

for model in df.index:
    plt.plot(df.columns, df.loc[model], marker='o', label=model)

plt.xlabel('Dataset')
plt.ylabel('AUC-ROC (%)')
# plt.title('Model Performance with Different Across Different Datasets')
plt.title('')
plt.xticks(df.columns)  # Ensure only DS1 to DS6 are shown as x labels
plt.yticks(range(10, 101, 10))  # Set y ticks from 10% to 100%
plt.legend(title='Models')
plt.grid(True)
plt.show()
