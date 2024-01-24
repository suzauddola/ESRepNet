import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import sys
# import os

# # Add the root directory of the repository to the Python path
# script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
# root_dir = os.path.join(script_dir, '..', '..')  # Root directory of the repo
# sys.path.append(root_dir)

from models.yolo import Model
from utils.general import non_max_suppression, scale_coords
# from utils.augmentations import letterbox
from utils.datasets import letterbox

import torch
import cv2
import numpy as np


# def save_feat(module, input, output):
#     global activations, gradients
#     activations = output
#     def hook_gradients(grad):
#         global gradients
#         gradients = grad
#     output.register_hook(hook_gradients)

# def get_heatmap(model, layer_name, img, class_idx):
#     # Function to extract activations and gradients

#    for name, module in model.named_modules():
#     print(name, module)


#     # Define hook for activations
#     activations = None
#     gradients = None

#     # Register hook to the layer
#     layer = dict(model.named_modules())[layer_name]


#     # Register the hook to the SelfAttention layer within model.50
#     self_attention_layer = layer.selfAttention[0]
#     hook_handle = self_attention_layer.register_forward_hook(save_feat)
   
#     # # Register the hook
#     # hook_handle = layer[0].register_forward_hook(save_feat)

#     # Forward pass
#     pred = model(img)

#     # Select the class score for the given class index
#     # class_score = pred[0, class_idx, 4] * pred[0, class_idx, 5 + class_idx]

#     first_tensor = pred[0].cpu().detach().numpy()
#     class_score = first_tensor[0, class_idx, 4] * first_tensor[0, class_idx, 5 + class_idx]

#     # Select the class score for the given class index
#     # class_score = pred[0, class_idx, 4] * pred[0, class_idx, 5 + class_idx]

#     class_score = class_score.sum()  # Sum to create a scalar quantity

#     # Convert class_score to a PyTorch tensor
#     class_score = torch.tensor(class_score, requires_grad=True).to('cuda')

#     # Zero the gradients
#     model.zero_grad()

#     # Calculate gradients of the class score with respect to the activations
#     class_score.backward(retain_graph=True)

#     # Print activations and gradients for debugging
#     print("Activations:", activations)
#     print("Gradients:", gradients)

#     # # Get the gradients from the SelfAttention layer
#     # gradients = gradients[0]  # Assuming gradients is a list with one element

#     # # Calculate the Grad-CAM heatmap
#     # heatmap = torch.mean(gradients, dim=[1, 2], keepdim=True)
#     # heatmap = torch.relu(heatmap)  # ReLU to remove negative values
#     # heatmap = heatmap.squeeze().cpu().numpy()

#     # # Resize heatmap to match input image
#     # heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
#     # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)

#      # Get the gradients from the hook
#     gradients = gradients[0] if gradients is not None else None  # Assuming gradients is a list with one element

#     if gradients is not None:
#         # Calculate the Grad-CAM heatmap
#         heatmap = torch.mean(gradients, dim=[1, 2], keepdim=True)
#         heatmap = torch.relu(heatmap)  # ReLU to remove negative values
#         heatmap = heatmap.squeeze().cpu().numpy()

#         # Resize heatmap to match input image
#         heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
#         heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
#     else:
#         heatmap = None

#     # Remove the hook
#     hook_handle.remove()

#     return heatmap

def get_heatmap(model, layer_name, img, class_idx):
    global activations, gradients  # Declare these variables as global

    activations = None
    gradients = None

    # Function to extract activations and gradients
    def save_grad(grad):
        global gradients
        gradients = grad

    # Define hook for activations
    # activations = {}

 # Function to extract activations and gradients
    def hook_gradients(grad):
        global gradients
        gradients = grad

    # Define hook for activations
    def hook_activations(module, input, output):
        global activations
        activations = output
        output.register_hook(hook_gradients)

    # Register hook to the layer
    layer = dict(model.named_modules())[layer_name]
    hook_handle = layer.register_forward_hook(hook_activations)
   
    # Forward pass
    pred = model(img)

    # After the forward pass, remove the hook
    hook_handle.remove()

    # Select class and backward pass
    # class_score = pred[0, class_idx, 4] * pred[0, class_idx, 5 + class_idx]


    first_tensor = pred[0].cpu().detach().numpy()
    class_score = first_tensor[0, class_idx, 4] * first_tensor[0, class_idx, 5 + class_idx]

    # Select the class score for the given class index
    # class_score = pred[0, class_idx, 4] * pred[0, class_idx, 5 + class_idx]

    class_score = class_score.sum()  # Sum to create a scalar quantity

    # Convert class_score to a PyTorch tensor
    class_score = torch.tensor(class_score, requires_grad=True).to('cuda')


    model.zero_grad()
    class_score.backward()

    # Check if gradients are captured
    if gradients is None:
        raise RuntimeError("Failed to capture gradients. Backward pass might not be working as expected.")

    # Calculate Grad-CAM
    gradients_mean = torch.mean(gradients, dim=[2, 3], keepdim=True)
    activations = torch.nn.functional.relu((activations * gradients_mean).sum(dim=1)).squeeze()

    # Resize heatmap to match input image
    heatmap = activations.cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    return heatmap

model_yaml_path = '/home/suza/YOLO/yolov7/cfg/mytrian/v7-SA-beforeSPP.yaml'


# Assuming 'model_pt_path' is the path to your saved weights file.
# model_pt_path = 'path/to/yolov7.pt'
model_pt_path = '/home/suza/YOLO/yolov7/runs/train/v7-mosic-6-pretraind-SAbeforeSPP/weights/best.pt'

# Load model
# model = Model('path/to/yolov7.yaml', ch=3, nc=80)  # Update path and parameters
model = Model(model_yaml_path , ch=3, nc=80)  # Update path and parameters
# model.load_state_dict(torch.load('path/to/yolov7.pt')['model'])  # Update path
# model.load_state_dict(torch.load(model_pt_path)['model'])  # Update path

# Load the model weights
checkpoint = torch.load(model_pt_path)

# Check if the loaded checkpoint contains a 'model' key
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    # If 'model' is a key, then the actual model might be stored under this key
    model_data = checkpoint['model']
    
    # Check if the 'model' key points to a state_dict or a model instance
    if isinstance(model_data, dict):
        # If 'model' key points to a state_dict, load it
        model.load_state_dict(model_data)
    elif isinstance(model_data, torch.nn.Module):
        # If 'model' key points to a model instance, use it directly
        model = model_data
    else:
        raise TypeError("The 'model' key within the checkpoint does not point to a state_dict or a model instance.")
else:
    raise TypeError("The checkpoint does not contain a 'model' key or the checkpoint structure is not recognized.")


# This ensures the model is in evaluation mode
model.eval()

# Convert the model to the appropriate precision and device
model.to('cuda').float()  # or model.to('cuda').half() if using half precision

# Load and preprocess the image
img_path = '/home/suza/YOLO/yolov7/TTOP_basic/test/images/0090_jpg.rf.5f2ba82f81205438109882f33f4953fb.jpg'
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Preprocess the image
img = letterbox(original_img, new_shape=640)[0]
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).float()
img /= 255.0
img = img.unsqueeze(0)

# Move to the same device as the model
img = img.to('cuda')

# Set model to training mode (gradient tracking)
model.train()

# ... code to load and preprocess your image ...

# Enable gradient tracking for the input image tensor
img.requires_grad_(True)

# Set the class index for which you want to generate the heatmap
# Replace '0' with the actual class index you're interested in
class_idx = 0

# ... rest of your code ...
import numpy as np

# Forward pass
pred = model(img)
# print(pred_tuple)

# for i, tensor in enumerate(pred):
#     print(f"Tensor {i}: shape = {tensor.shape}")

# pred = np.array(pred)
# pred = Tensor.cpu().numpy()



# Assuming the predictions are the first element of the tuple
# detections  = pred[0]

# detections = detections.cpu().detach().numpy()

# Access the class score using the class index
# Ensure the class score is a scalar by summing up
# class_score = detections[0, class_idx, 4] * detections[0, class_idx, 5 + class_idx]
# class_score = pred[0][class_idx][4] * pred[0][class_idx][5 + class_idx]

# Assuming pred is a list of tensors, and you want to access the first tensor in the list
first_tensor = pred[0].cpu().detach().numpy()
class_score = first_tensor[0, class_idx, 4] * first_tensor[0, class_idx, 5 + class_idx]

class_score = class_score.sum()  # Sum to create a scalar quantity

# Convert class_score to a PyTorch tensor
class_score = torch.tensor(class_score, requires_grad=True).to('cuda')


# Backward pass to get gradients
model.zero_grad()
class_score.backward()

# Now continue with your heatmap generation code...
# Get heatmap
heatmap = get_heatmap(model, 'model.75', img, class_idx=0)

# Code for overlaying the heatmap on the original image continues here...

# Overlay heatmap on original image
overlay_img = original_img.copy()
overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay_img = cv2.addWeighted(overlay_img, 0.6, heatmap_img, 0.4, 0)

# Display image
cv2.imshow('Grad-CAM', overlay_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
