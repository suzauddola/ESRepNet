# Plotting Class Loss and Precision together



import matplotlib.pyplot as plt

# Converting 'Epochs' to numeric values for plotting
data['Epoch'] = data['Epochs'].str.split('/').str[0].astype(int)

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(12, 20))

# Losses
axs[0].plot(data['Epoch'], data['box_loss'], label='Box Loss')
axs[0].plot(data['Epoch'], data['obj_loss'], label='Object Loss')
axs[0].plot(data['Epoch'], data['class_loss'], label='Class Loss')
axs[0].set_title('Losses over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Precision and Recall
axs[1].plot(data['Epoch'], data['Precision'], label='Precision')
axs[1].plot(data['Epoch'], data['Recal'], label='Recall')
axs[1].set_title('Precision and Recall over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Value')
axs[1].legend()

# mAP at IoU threshold 0.5
axs[2].plot(data['Epoch'], data['mAP0.5'], label='mAP@0.5')
axs[2].set_title('mAP@0.5 over Epochs')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('mAP')

# mAP at IoU threshold 0.95
axs[3].plot(data['Epoch'], data['mAP0.95'], label='mAP@0.95')
axs[3].set_title('mAP@0.95 over Epochs')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('mAP')

# GPU Usage
axs[4].plot(data['Epoch'], data['GPU'].str[:-1].astype(float), label='GPU Usage (G)')
axs[4].set_title('GPU Usage over Epochs')
axs[4].set_xlabel('Epoch')
axs[4].set_ylabel('GPU Usage (G)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Class Loss
plt.plot(data['Epoch'], data['class_loss'], label='Class Loss', color='blue')

# Precision
plt.plot(data['Epoch'], data['Precision'], label='Precision', color='orange')

plt.title('Class Loss and Precision over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()