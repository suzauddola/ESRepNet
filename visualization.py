import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigh_path = './runs/train/nv7-mosic-15-pretraind/weights/best.pt'
# weigh_path = './runs/train/v7-mosic-6-pretraind-SAbeforeSPP/weights/best.pt'
# weigths = torch.load('./weights/yolov7-e6e.pt')
weigths = torch.load(weigh_path)
model = weigths['model']
model = model.half().to(device)
_ = model.eval()
# /home/suza/YOLO/yolov7/only_mosic/test/images/36_9_jpg.rf.839b757cdb3c574306468d057742551c.jpg
# img_path = './v1.jpg'
img_path = './TTOP_basic/valid/images/31_7_jpg.rf.82febebd92d6b9a574856d8032c98556.jpg'
# image = cv2.imread('./inference/images/person.png')  # 504x378 image
image = cv2.imread(img_path)  # 504x378 image
# image = letterbox(image, 1280, stride=64, auto=True)[0]
image = letterbox(image, 640, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
image = image.to(device)
image = image.half()

output = model(image)

obj1 = output[1][0][0, 0, :, :, 4].detach().sigmoid().cpu().numpy()
obj2 = output[1][0][0, 1, :, :, 4].detach().sigmoid().cpu().numpy()
obj3 = output[1][0][0, 2, :, :, 4].detach().sigmoid().cpu().numpy()
obj4 = output[1][1][0, 0, :, :, 4].detach().sigmoid().cpu().numpy()
obj5 = output[1][1][0, 1, :, :, 4].detach().sigmoid().cpu().numpy()
obj6 = output[1][1][0, 2, :, :, 4].detach().sigmoid().cpu().numpy()
obj7 = output[1][2][0, 0, :, :, 4].detach().sigmoid().cpu().numpy()
obj8 = output[1][2][0, 1, :, :, 4].detach().sigmoid().cpu().numpy()
obj9 = output[1][2][0, 2, :, :, 4].detach().sigmoid().cpu().numpy()
# obj10 = output[1][3][0, 0, :, :, 4].detach().sigmoid().cpu().numpy()
# obj11 = output[1][3][0, 1, :, :, 4].detach().sigmoid().cpu().numpy()
# obj12 = output[1][3][0, 2, :, :, 4].detach().sigmoid().cpu().numpy()


# %matplotlib inline
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(image_[:,:,[2,1,0]])
plt.show()
fig, ax = plt.subplots(3,3,figsize=(16,12))
#[ax_.axis('off') for ax_ in ax.ravel()]
[ax_.set_xticklabels([]) for ax_ in ax.ravel()]
[ax_.set_yticklabels([]) for ax_ in ax.ravel()]
ax.ravel()[0].imshow(obj1)
ax.ravel()[1].imshow(obj2)
ax.ravel()[2].imshow(obj3)
ax.ravel()[3].imshow(obj4)
ax.ravel()[4].imshow(obj5)
ax.ravel()[5].imshow(obj6)
ax.ravel()[6].imshow(obj7)
ax.ravel()[7].imshow(obj8)
ax.ravel()[8].imshow(obj9)
# ax.ravel()[9].imshow(obj10)
# ax.ravel()[10].imshow(obj11)
# ax.ravel()[11].imshow(obj12)
plt.subplots_adjust(wspace=-0.52, hspace=0)
plt.show()

# # Save the figure with a higher DPI
high_res_output_path = '/home/suza/YOLO/yolov7/31_7_jpg.png'
plt.savefig(high_res_output_path, dpi=400)

