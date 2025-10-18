import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from utils import collate_fn_predict, OpenCVResizer

# 定义图像预处理的变换操作
img_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB") if x.mode != 'RGB' else x),  # 如果不是RGB，转换为RGB
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
])

# 掩码只需要转换为二值Tensor
def mask_to_tensor(mask):
    # 将PIL图像转换为numpy数组，然后转换为Tensor
    mask_np = torch.from_numpy(np.array(mask))
    # 将非零值(255)转换为1，保持0为0
    mask_tensor = (mask_np > 0).float()
    return mask_tensor
    
class TestDataset_for_DETR(Dataset):
    def __init__(self, imgdir_path, target_size=(800, 800)):
        super().__init__()
        self.img_lst = []
        self.target_size = target_size

        self.resizer = OpenCVResizer(target_size=target_size)

        self.img_lst = [os.path.join(imgdir_path, i) for i in os.listdir(imgdir_path) ]

        # 预缓存图像路径映射 (可选优化)
        self.img_cache = {}
        print(f"Dataset initialization completed, a total of {len(self.img_lst)} samples")

    def __len__(self):
        return len(self.img_lst)   
    
    def __getitem__(self, index):
        # 处理图像和掩码
        resizer = self.resizer(self.img_lst[index])
        resized_img, mask = resizer  # 获取填充后的图像和掩码

        # 分别转换图像和掩码
        input_img = img_transform(resized_img)
        # 将掩码转换为二值张量 (0和1)
        mask_tensor = mask_to_tensor(mask).squeeze(0)  # 从(1,H,W)变为(H,W)
        
        return input_img, mask_tensor

if __name__ == '__main__':
    # tw=800, th=800
    imgsz = (800, 800)
    batch_size = 4

    imgdir_path = "your_img_path"

    mydataset = TestDataset_for_DETR(imgdir_path, target_size=imgsz)
    mydataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_predict, num_workers=0, pin_memory=True)

    for batch in mydataloader:
        imgs, masks = batch
        print(f"图像尺寸: {imgs.shape}")          # torch.Size([bs, 3, H, W])
        print(f"掩码尺寸: {masks.shape}")         # torch.Size([bs, 800, 800]) 
        break