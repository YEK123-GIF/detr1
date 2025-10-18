import os
import cv2
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from utils import collate_fn_train_val

# 适合于Linux系统, 使用多线程workers, window线程只使用workers=0
class OpenCVResizer:
    def __init__(self, target_size=(800, 800)):
        self.tw, self.th = target_size

    def __call__(self, img_path):
        """
        使用OpenCV加速图像缩放和填充
        :param img_path: 图像路径
        :return: 处理后的图像(PIL), 掩码(PIL), 原始尺寸(tuple)
        """
        # 使用OpenCV读取图像 (比PIL快2-3倍)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转换为RGB格式, RGB是PIL默认的格式
        
        h, w = img.shape[:2] # img.shape--> [h, w, channel]
        
        # 缩放到指定imgsz
        scale_w = self.tw / w
        scale_h = self.th / h
        scale = min(scale_w, scale_h) #保证new_w(new_h)不大于tw(th)

        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用OpenCV加速缩放 (比PIL快3-5倍)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 创建新图像和掩码
        new_img = np.zeros((self.th, self.tw, 3), dtype=np.uint8)
        mask = np.ones((self.th, self.tw), dtype=np.uint8) * 255
        
        # 将缩放后的图像放入左上角
        new_img[0:new_h, 0:new_w] = resized_img
        mask[0:new_h, 0:new_w] = 0
        
        # 转换为PIL格式保持兼容性
        return Image.fromarray(new_img), Image.fromarray(mask), (new_w, new_h)

# 定义图像预处理的变换操作
img_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != 'RGB' else x),  # 如果不是RGB，转换为RGB
    transforms.ToTensor(),  # 将图像转换为Tensor #默认 1. (H, W, C) -> (C, H, W) 2.像素值[0, 255]归一化到[0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
])

# 掩码只需要转换为二值Tensor
def mask_to_tensor(mask):
    # 将PIL图像转换为numpy数组，然后转换为Tensor
    mask_np = torch.from_numpy(np.array(mask))
    # 将非零值(255)转换为1，保持0为0
    mask_tensor = (mask_np > 0).float()
    return mask_tensor

def process_txtdata(txt_path):
    with open(txt_path, 'r') as f:
        content = f.readlines()
        boxes = []
        for i in content:
            line = i.strip()
            parts = line.split()# 去除空格
            new_part = [int(parts[x]) if x==0  else float(parts[x])  for x in range(len(parts))]
            boxes.append(deepcopy(new_part))
        boxes = torch.tensor(boxes) # [num_bbox, 5], 第一维度为一张图片中的框数量, 第二维度为每个框的[id, xc, yc, w, h]的归一化数据
        box_ids = boxes[:, 0].long() # tensor(1,0,1,2...) (num_bbox,)
        box_xywhs = boxes[:, 1:] # (num_bbox, 4)
        return box_ids, box_xywhs
    
class TrainDataset_for_DETR(Dataset):
    def __init__(self, imgdir_path, txtdir_path, target_size=(800, 800)):
        super().__init__()
        self.img_lst = []
        self.txt_lst = []
        self.target_size = target_size
        
        # 使用OpenCV加速器
        self.resizer = OpenCVResizer(target_size=target_size)
        
        # 构建图像和标签路径列表
        self.img_lst = [os.path.join(imgdir_path, f) for f in os.listdir(imgdir_path)]
        self.txt_lst = [os.path.join(txtdir_path, f) for f in os.listdir(txtdir_path)]
        
        # 确保图像和标签匹配
        self.img_lst.sort()
        self.txt_lst.sort()
        assert len(self.img_lst) == len(self.txt_lst), "图像和标签数量不匹配"
        
        # 预缓存图像路径映射 (可选优化)
        self.img_cache = {}
        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")

    def __len__(self):
        return len(self.img_lst)   
    
    def __getitem__(self, index):
        # 处理图像和掩码
        resizer = self.resizer(self.img_lst[index])
        resized_img, mask, w1h1 = resizer  # 获取填充后的图像和掩码
        
        # 分别转换图像和掩码
        input_img = img_transform(resized_img)
        # 将掩码转换为二值张量 (0和1)
        mask_tensor = mask_to_tensor(mask).squeeze(0)  # 从(1,H,W)变为(H,W)
        
        # 处理标签数据
        real_id, real_bbox = process_txtdata(self.txt_lst[index])
        w1, h1 = w1h1
        # real_bbox归一化到 tw, th大小上
        real_bbox[:, 0] *= w1 / self.target_size[0]  # 第 0 列乘以 w1/tw
        real_bbox[:, 1] *= h1 / self.target_size[1]  # 第 1 列乘以 h1/th
        real_bbox[:, 2] *= w1 / self.target_size[0]  # 第 2 列乘以 w1/tw
        real_bbox[:, 3] *= h1 / self.target_size[1]  # 第 3 列乘以 h1/th
        # real_bbox不仅适用于xywh格式, 也适用于xyxy格式, 调整到指定目标大小下的归一化坐标
        return input_img, mask_tensor, real_id, real_bbox, w1h1

# 演示
if __name__ == '__main__':
    # imgsz = (tw, th)
    imgsz = (640, 640)
    batch_size = 4

    imgdir_path = "your_img_path"
    txtdir_path = "your_txt_path"

    mydataset = TrainDataset_for_DETR(imgdir_path, txtdir_path, target_size=imgsz)

    mydataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train_val, num_workers=0, pin_memory=True) #pin_memory=True 表示将数据加载到 锁页内存（pinned memory，也叫 page-locked memory）
    # pin_memory=True 表示将数据加载到 锁页内存（pinned memory，也叫 page-locked memory）
    # 它不能被操作系统换出到磁盘，因此 GPU 可以 直接通过 DMA（Direct Memory Access）从主机内存读取数据，从而加快从 CPU 到 GPU 的数据拷贝速度。
    #num_workers 控制了 用于加载数据的子进程数量
    # (bs, m, n)---x    (bs,[tensor1, tensor2...])
    for batch in mydataloader:
        imgs, masks, real_id, real_bbox, w1h1 = batch
        print(f"图像尺寸: {imgs.shape}")          # torch.Size([bs, 3, H, W])
        print(f"掩码尺寸: {masks.shape}")         # torch.Size([bs, 800, 800])
        print(f"real_id: {len(real_id)}")
        print(f"real_id: {real_id}")
        print(f"real_bbox: {len(real_bbox)}")    
        print(f"w1h1: {w1h1}")  # w1h1: [[640, 480], [640, 426], [640, 428], [640, 425]]
        break