import torch
from torchvision.ops import box_convert

import cv2
import numpy as np
from PIL import Image

import os 

def xywh_to_xyxy(xywh):
    if not isinstance(xywh, torch.Tensor):
        xyxy = [box_convert(i, in_fmt="cxcywh", out_fmt="xyxy") for i in xywh]
        return xyxy
    xyxy = box_convert(xywh, in_fmt="cxcywh", out_fmt="xyxy")
    return xyxy
        
# 自定义collate_fn_train_val处理变长训练验证数据
def collate_fn_train_val(batch):
    images, masks, ids_list, bboxes_list, w1h1 = zip(*batch)
    
    # 堆叠图像和掩码
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    # 处理变长的标签数据（不堆叠）
    return images, masks, ids_list, bboxes_list, w1h1

# 自定义collate_fn_predict处理变长测试数据
def collate_fn_predict(batch):
    images, masks = zip(*batch)
    
    # 堆叠图像和掩码
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    # 处理变长的标签数据（不堆叠）
    return images, masks


def per_img_gt(txtdir):
    name_lst = os.listdir(txtdir)
    count = 0
    list_len = []
    for name in name_lst:
        with open(os.path.join(txtdir, name), 'r') as f:
            content = f.readlines()
            count = count + len(content)
            list_len.append(len(content))
    max_len = max(list_len)
    average_gt = count/len(name_lst)
    return average_gt, max_len

def caculate_num_queries(txtdir, background_weight=0.1):
    # 设定背景权重为0.1时, 推荐的自适应queries数量
    average_gt, max_len = per_img_gt(txtdir)
    background_weight = 0.1
    caculate_queries = average_gt * (1 + 1 / background_weight)
    proposal_num_queries = max(int(caculate_queries), max_len)
    
    return proposal_num_queries

def caculate_background_weight(txtdir, num_queries=100):
    # 设定100个queries时候,推荐的自适应背景权重
    average_gt, _ = per_img_gt(txtdir)
    background_weight = average_gt*1 / (num_queries-average_gt)

    return background_weight

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        
        h, w = img.shape[:2]

        # 缩放到指定大小
        scale_w = self.tw / w
        scale_h = self.th / h
        scale = min(scale_w, scale_h)

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
        return Image.fromarray(new_img), Image.fromarray(mask)