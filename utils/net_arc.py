
import matplotlib.pyplot as plt

# Creating a simplified flowchart for the re-parameterization technique in YOLOv7

fig, ax = plt.subplots(figsize=(10, 6))

# Nodes representing different steps in the re-parameterization process
steps = [
    "Standard Convolution Layer",
    "Decompose into Depthwise & Pointwise",
    "Add Non-linearity (e.g., SiLU/Swish)",
    "Apply Skip Connections",
    "Use Batch Normalization",
    "Fusion of Reparameterized Layers",
    "Output to Next Layers"
]

# Plotting the nodes
for i, step in enumerate(steps):
    ax.text(0.1, 1 - 0.15 * i, step, fontsize=12, bbox=dict(facecolor='orange', alpha=0.5))

# Styling
ax.set_title("Re-parameterization Technique in YOLOv7", fontsize=16)
ax.axis('off')

plt.show()
high_res_output_path = '/home/suza/YOLO/yolov7/net_arc.png'
plt.savefig(high_res_output_path, dpi=400)
