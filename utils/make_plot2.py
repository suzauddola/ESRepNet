import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text

# Load the dataset
df = pd.read_csv('/home/suza/YOLO/yolov7/precision13456.csv')
# df = '/home/suza/YOLO/yolov7/precision13456.csv'
# Since the high-resolution plot generation with adjustments is causing issues, let's attempt to generate a standard resolution plot.

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Re-loading the dataset
# df = pd.read_csv('/mnt/data/precision13456.csv')

# Convert to long format
df_long = df.melt(id_vars='Iteration', var_name='group', value_name='model')

# Sort and extract the last point for each group
last_points = df_long.sort_values(by=['group', 'Iteration']).groupby('group').tail(1)

# Initialize the figure
plt.figure(figsize=(12, 6))

# Create the line plot
sns.lineplot(data=df_long, x='Iteration', y='model', hue='group', 
             palette=['red', 'green', 'blue', 'orange', 'purple'])

# Plot the last points with larger dots
sns.scatterplot(data=last_points, x='Iteration', y='model', hue='group', 
                palette=['red', 'green', 'blue', 'orange', 'purple'], 
                legend=False, s=100, zorder=5)

# Annotate the last points with their respective values
for _, row in last_points.iterrows():
    plt.text(row['Iteration'], row['model'], 
             f"{row['model']:.4f}", color='black', ha="center", va="center", fontsize=8)

# Title and labels
plt.title('Model Precision Over Iterations', fontsize=14, weight='bold')
plt.xlabel('Iteration', fontsize=12, weight='bold')
plt.ylabel('Precision', fontsize=12, weight='bold')

# Axes limits
plt.xlim(304556, 306700)
plt.ylim(0.92, 1)

# # Show the plot
# plt.show()



# Save the plot
plt.savefig('/home/suza/YOLO/yolov7/high_res_plot.png', dpi=400)
# plt.show()
