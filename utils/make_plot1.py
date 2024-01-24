# Let's address the overlap issue with a different approach. We'll try to manually set the y-offsets for overlapping annotations.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset again
file_path = '/home/suza/YOLO/yolov7/precision_99.csv'
df = pd.read_csv(file_path)

# Convert to long format
df_long = df.melt(id_vars='Iteration', var_name='group', value_name='model')

# Sort and extract the last point for each group
last_points = df_long.sort_values(by=['group', 'Iteration']).groupby('group').tail(1)

# Initialize the figure
plt.figure(figsize=(14, 8), dpi=300)

# Create the line plot
sns.lineplot(data=df_long, x='Iteration', y='model', hue='group', 
             palette=['red', 'green', 'blue', 'orange', 'purple'])

# Plot the last points with larger dots
sns.scatterplot(data=last_points, x='Iteration', y='model', hue='group', 
                palette=['red', 'green', 'blue', 'orange', 'purple'], 
                legend=False, s=100, zorder=5)

# Manually adjust the y-position for each label to avoid overlaps
y_offsets = {'ESRepNet': 0.0030, 'MSFF': -0.0030, 'ESAttM': 0.0030, 'ENST': 0.0030, 'Base Model': -0.0030}

# Annotate the last points with their respective values
for _, row in last_points.iterrows():
    # Apply manual y-offsets
    y_offset = y_offsets.get(row['group'], 0)
    plt.text(row['Iteration'], round(row['model'], 5) + y_offset, # ,row['model'
             f"{row['model']:.4f}", ha="center", va="center", fontsize=16,  weight='semibold')

# Title and labels
plt.title('Model Precision Over Iterations', fontsize=16, weight='bold')
plt.xlabel('Iteration', fontsize=16, weight='bold')
plt.ylabel('Precision', fontsize=16, weight='bold')

# Legend
plt.legend(title='', fontsize=14, loc='lower left') #,  bbox_to_anchor=(0.0,0.7))# loc='best', )

# Axes limits
plt.xlim(10850, 11010)
plt.ylim(0.900, 1)
plt.grid(True)

# Save the figure with a higher DPI
high_res_output_path = '/home/suza/YOLO/yolov7/make_plot1.png'
plt.savefig(high_res_output_path, dpi=400)

# Show the plot
# plt.show()

# Return the path to the saved high-resolution plot
# output_path

