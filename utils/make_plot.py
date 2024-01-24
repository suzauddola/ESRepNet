# Since the previous attempts to create a high-resolution plot programmatically were unsuccessful,
# let's try one more time to create a high-resolution plot that fixes the overlapping issue and is downloadable.

# First, we'll re-import the libraries and the data since we encountered an exception before.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset again
file_path = '/home/suza/YOLO/yolov7/precision13456.csv'
# /home/suza/YOLO/yolov7/precision13456.csv
df = pd.read_csv(file_path)

# Convert to long format
df_long = df.melt(id_vars='Iteration', var_name='group', value_name='model')

# Sort and extract the last point for each group
last_points = df_long.sort_values(by=['group', 'Iteration']).groupby('group').tail(1)

# Create a larger plot to give space for annotations
plt.figure(figsize=(12, 6), dpi=150)

# Plot all lines
line_plot = sns.lineplot(data=df_long, x='Iteration', y='model', hue='group', palette='deep')

# Highlight the last points
sns.scatterplot(data=last_points, x='Iteration', y='model', hue='group', palette='deep', legend=False, s=50, zorder=5)

# Offset for text annotations
offset = (last_points['Iteration'].max() - last_points['Iteration'].min()) * 0.8


# Add text annotations
for _, row in last_points.iterrows():
    plt.text(row['Iteration'] + offset, row['model'], 
             f"{row['model']:.4f}", fontsize=10, color='black', ha='left', va='center')


# Customize the plot
plt.title('Model Precision Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Precision')
plt.grid(True)
plt.legend(title='Model', loc='upper left')

# Set the x and y limits
plt.xlim(304556, 306700)
plt.ylim(0.9, 1.0)

# Tight layout for saving
plt.tight_layout()


# # Create a high-resolution plot
# plt.figure(figsize=(14, 8), dpi=300)
# line_plot = sns.lineplot(data=df_long, x='Iteration', y='model', hue='group', palette=['red', 'green', 'blue', 'orange', 'purple'])
# scatter_plot = sns.scatterplot(data=last_points, x='Iteration', y='model', hue='group', 
#                                palette=['red', 'green', 'blue', 'orange', 'purple'], 
#                                legend=False, s=100, zorder=5)

# # Annotate the last points with their respective values
# for _, row in last_points.iterrows():
#     plt.text(row['Iteration'], row['model'], 
#              f"{row['model']:.8f}", color='black', ha="left", va="bottom", fontsize=12)

# # Enhance the plot
# plt.title('Model Precision Over Iterations', fontsize=16, weight='bold')
# plt.xlabel('Iteration', fontsize=12, weight='bold')
# plt.ylabel('Precision', fontsize=12, weight='bold')
# # plt.legend(title='', fontsize=14, loc='best')

# # Set the axis limits
# iteration_min, iteration_max = 304556, 306700
# plt.xlim(iteration_min, iteration_max)
# plt.ylim(0.92, 1)
# plt.legend(title='', loc='center left', fontsize=12)

# Tight layout for better spacing
# plt.tight_layout()

# Save the plot with a high DPI
high_res_output_path = '/home/suza/YOLO/yolov7/high_res_precision_plot.png'
plt.savefig(high_res_output_path, dpi=400)

# Show the plot
# plt.show()

# Return the path to the saved high-resolution plot
# high_res_output_path

