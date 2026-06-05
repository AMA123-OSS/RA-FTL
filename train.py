import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 可选：限制线程数
os.environ['OMP_NUM_THREADS'] = '1'
import torch
torch.backends.cudnn.enabled = False
from torch import nn,optim
from torch.utils.data import DataLoader
from data import *
#from net import * #unet用
from FCN import *
# from FCN import *
from torchvision.utils import save_image

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

VOC_PALETTE = [
    (0, 0, 0),        # 0: background
    (128, 0, 0),     # 1: aeroplane
    (0, 128, 0),     # 2: bicycle
    (128, 128, 0),   # 3: bird
    (0, 0, 128),     # 4: boat
    (128, 0, 128),   # 5: bottle
    (0, 128, 128),   # 6: bus
    (128, 128, 128), # 7: car
    (64, 0, 0),      # 8: cat
    (192, 0, 0),     # 9: chair
    (64, 128, 0),    # 10: cow
    (192, 128, 0),   # 11: dining table
    (64, 0, 128),    # 12: dog
    (192, 0, 128),   # 13: horse
    (64, 128, 128),  # 14: motorbike
    (192, 128, 128), # 15: person
    (0, 64, 0),      # 16: potted plant
    (128, 64, 0),    # 17: sheep
    (0, 192, 0),     # 18: sofa
    (128, 192, 0),   # 19: train
    (0, 64, 128)     # 20: tv/monitor
]


import numpy as np
import torch

def seg_to_palette(seg):
    """
    seg: torch.Tensor [H, W], values in [0,20] or 255
    return: torch.Tensor [3, H, W], RGB, uint8
    """
    device = seg.device  # ✅ 记住输入设备

    if seg.dim() == 3:
        seg = seg.squeeze(0)
    if seg.dim() != 2:
        raise ValueError(f"seg must be [H, W], got {seg.shape}")

    seg = seg.cpu().numpy()
    h, w = seg.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(VOC_PALETTE):
        color_mask[seg == cls_id] = color

    color_mask[seg == 255] = (0, 0, 0)

    # ✅ 放回原 device
    return torch.from_numpy(color_mask).permute(2, 0, 1).to(device)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path= 'params/unet.pth'
data_path=r'D:\Users\HP\Desktop\shixun\VOC数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'

save_path=r'.\train_image'#保存训练结果
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=8,shuffle=True)
    # net=UNet().to(device)
    net = FCN8s(num_classes=21).to(device)

    # print(net)
    # if os.path.exists(weight_path):
    #     net.load_state_dict(torch.load(weight_path))
    #     print('successful load weight！')
    # else:
    #     print('not successful load weight')

    opt=optim.Adam(net.parameters(), lr=1e-4)
    # loss_fun=nn.BCELoss()

    weights = torch.ones(21, device=device)
    weights[0] = 0.1  # 背景
    weights[15] = 1.0  # person
    weights[1:] = 3.0  # 小类


    loss_fun = nn.CrossEntropyLoss(
        ignore_index=255,
        weight=weights
    )

    for epoch in range(2000):
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)

            out_image=net(image)

            train_loss=loss_fun(out_image,segment_image)# segment_image: ground_truth
            # print(segment_image.shape)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%16==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%16==0:
                torch.save(net.state_dict(),weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0].argmax(dim=0)


            gt_color = seg_to_palette(_segment_image)
            pred_color = seg_to_palette(_out_image)

            img = torch.stack([_image, gt_color, pred_color], dim=0)


            save_image(img,f'{save_path}/{i}.png')

        # epoch+=1


