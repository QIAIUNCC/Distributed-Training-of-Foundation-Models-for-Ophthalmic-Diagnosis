import pandas as pd
import matplotlib.pyplot as plt

# Data from the table
data = {
    'Dataset': ['DS1', 'DS1', 'DS2', 'DS2', 'DS3', 'DS3', 'DS4', 'DS4', 'DS5', 'DS5', 'DS6', 'DS6'],
    'Model': ['128x128', '224x224', '128x128', '224x224', '128x128', '224x224', '128x128', '224x224',
              '128x128', '224x224', '128x128', '224x224'],
    'Test-1': [84.68, 87.02, 63.97, 67.22, 33.61, 35.76, 68.14, 64.67, 21.71, 21.92, 53.85, 54.65],
    'Test-2': [88.68, 96.37, 84.96, 93.42, 76.09, 75.37, 60.76, 78.32, 48.88, 40.41, 54.46, 55.3],
    'Test-3': [76.74, 92.46, 53.93, 59.12, 83.23, 87.27, 49.92, 72.22, 69.03, 71.78, 69.77, 70.97],
    'Test-4': [49.4, 52.37, 54.88, 54.12, 24.93, 30.84, 67.96, 64.68, 22.5, 23.5, 61.49, 63.36],
    'Test-5': [85.87, 80.87, 48.13, 55.12, 69.25, 65.23, 57.75, 75.25, 72.8, 73.25, 20.13, 25.25],
    'Test-6': [66.21, 73.72, 52.88, 56.04, 66.5, 71.95, 50.68, 54.3, 43.65, 48.95, 59.8, 64.14],
    'Average': [75.26, 80.47, 59.79, 64.17, 58.94, 61.07, 59.20, 68.24, 46.43, 46.64, 53.25, 55.95]
}

# Create DataFrame
df = pd.DataFrame(data)

# Filter out the data for 128x128 and 224x224 models
df_128 = df[df['Model'] == '128x128']
df_224 = df[df['Model'] == '224x224']

# Setting the positions and width for the bars
pos = list(range(len(df_128['Average'])))
width = 0.35
plt.rcParams['font.family'] = 'Times New Roman'
# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 5))
plt.bar(pos, df_128['Average'], width, alpha=0.7, color='blue', label='128x128')
plt.bar([p + width for p in pos], df_224['Average'], width, alpha=0.7, color='red', label='224x224')

# Setting the y-axis label, chart title, and ticks
ax.set_ylabel('Average AUC-ROC')
# ax.set_title('Comparison of Model Performance (128x128 vs 224x224)')
ax.set_title('')
ax.set_xticks([p + 0.5 * width for p in pos])
ax.set_xticklabels(df_128['Dataset'])

# Adding the legend and grid
plt.legend(['128x128', '224x224'], loc='upper right')
plt.grid()
plt.show()
