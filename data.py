import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import  matplotlib.pyplot as plt


# ---------------------------
# 原图
# ---------------------------
def keep_image_size_open_rgb(path, size=(256, 256)):
    # img = Image.open(path).convert('RGB')#用open打开的可能还有其他模式的图像，如灰度图像
    img = Image.open(path)
    img = img.resize(size, Image.BILINEAR)
    return img

# ---------------------------
# VOC mask（✅ 修复 Pillow 报错）
# ---------------------------
def load_voc_mask(path, size=(256, 256)):
    mask = Image.open(path)
    # mask = np.array(mask, dtype=np.uint8)  # 转化成数组类型
    # mask = Image.fromarray(mask)  # 把 NumPy 数组转回 PIL 图像，方便做 resize
    mask = mask.resize(size, Image.NEAREST) # 把 mask 缩放到指定尺寸，最近邻插值，不修改类别值。
    return np.array(mask, dtype=np.int64) # 把缩放后的 mask 转成 int64 NumPy 数组，用于 PyTorch 训练。

# ---------------------------
# Transform
# ---------------------------
transform_img = transforms.Compose([
    transforms.ToTensor()
])

# ---------------------------
# Dataset
# ---------------------------
class MyDataset(Dataset):
    def __init__(self, path, size=(256, 256)):
        self.path = path
        self.size = size
        self.names = sorted(
            os.listdir(os.path.join(path, 'SegmentationClass'))
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        img_path = os.path.join(
            self.path, 'JPEGImages', name.replace('.png', '.jpg')
        )
        img = keep_image_size_open_rgb(img_path, self.size) # 原图像处理
        img = transform_img(img)

        seg_path = os.path.join(
            self.path, 'SegmentationClass', name
        )
        seg = load_voc_mask(seg_path, self.size) # ground truth 处理
        return img, seg

# ---------------------------
# 测试
# ---------------------------
if __name__ == '__main__':
    data = MyDataset(
        r'D:\Users\HP\Desktop\shixun\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    )

    img, seg = data[5]
    print('image:', img.shape)
    print('mask :', seg.shape)
    print('dtype:', seg.dtype)
    plt.imshow(seg)
    plt.show()
    print('values:', np.unique(seg))