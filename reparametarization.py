# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml

device = select_device('0', batch_size=1)
# model trained by cfg/training/*.yaml
# ckpt = torch.load('cfg/training/yolov7_training.pt', map_location=device)
# ckpt = torch.load('runs/train/nv7-basic-6-no_pretraind/weights/last.pt', map_location=device)
# /home/suza/YOLO/yolov7/runs/train/v7-mySPPC++/weights/last.pt

# ckpt = torch.load('runs/train/v7-mySPPC++/weights/last.pt', map_location=device)
# ckpt = torch.load('runs/train/nv7-mosic-15-pretraind/weights/last.pt', map_location=device)

ckpt = torch.load('runs/train/v7pest24MFCC1SAsppcsp4/weights/best.pt', map_location=device)

# ckpt = torch.load('weights/yolov7_training.pt', map_location=device)
# reparameterized model in cfg/deploy/*.yaml
# model = Model('cfg/deploy/yolov7.yaml', ch=3, nc=24).to(device)
model = Model('cfg/deploy/mySPP.yaml', ch=3, nc=24).to(device)

# with open('cfg/deploy/yolov7.yaml') as f:
with open('cfg/deploy/mySPP.yaml') as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'][0]) // 2

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

print(model)
print(model.nc)

# reparametrized YOLOR
for i in range((model.nc+5)*anchors):
    model.state_dict()['model.106.m.0.weight'].data[i, :, :, :] *= state_dict['model.106.im.0.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.106.m.1.weight'].data[i, :, :, :] *= state_dict['model.106.im.1.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.106.m.2.weight'].data[i, :, :, :] *= state_dict['model.106.im.2.implicit'].data[:, i, : :].squeeze()
model.state_dict()['model.106.m.0.bias'].data += state_dict['model.106.m.0.weight'].mul(state_dict['model.106.ia.0.implicit']).sum(1).squeeze()
model.state_dict()['model.106.m.1.bias'].data += state_dict['model.106.m.1.weight'].mul(state_dict['model.106.ia.1.implicit']).sum(1).squeeze()
model.state_dict()['model.106.m.2.bias'].data += state_dict['model.106.m.2.weight'].mul(state_dict['model.106.ia.2.implicit']).sum(1).squeeze()
model.state_dict()['model.106.m.0.bias'].data *= state_dict['model.106.im.0.implicit'].data.squeeze()
model.state_dict()['model.106.m.1.bias'].data *= state_dict['model.106.im.1.implicit'].data.squeeze()
model.state_dict()['model.106.m.2.bias'].data *= state_dict['model.106.im.2.implicit'].data.squeeze()

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
# torch.save(ckpt, 'weights/mySPPrep.pt')
torch.save(ckpt, 'runs/train/v7pest24MFCC1SAsppcsp4/weights/mySPPrep.pt')