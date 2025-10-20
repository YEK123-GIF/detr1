import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DetectionTransformer import DETR_model
from dataset_process_predict import TestDataset_for_DETR
from utils import xywh_to_xyxy, collate_fn_predict
from torchvision.ops import nms

# DETR模型预测
def predict_fn(imgdir, model_weights, device, output_dir=None, imgsz=(640, 640), conf_thres=0.7, 
               num_classes=80, num_queries=100, num_encoder_layer=6, num_decoder_layer=6, 
               batch_size=4, num_workers=0, pin_memory=True, NMS=(False, 0.5)):
    if output_dir != None:
        print(f"Pred [id, xmin, ymin, xmax, ymax]s write into {output_dir} folder, save as txt file.\n")
    else:
        print(f"Pred [id, xmin, ymin, xmax, ymax]s do not write into folder.\n")
    predict_dataset = TestDataset_for_DETR(imgdir, imgsz)
    predict_dataloder = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_predict,
                                   num_workers=num_workers, pin_memory=pin_memory)
    
    predict_model = DETR_model(num_queries=num_queries, num_classes=num_classes, 
                               num_encoder_layer=num_encoder_layer, num_decoder_layer=num_decoder_layer).to(device)
    predict_model.load_state_dict(torch.load(model_weights))
    predict_model.eval()

    with torch.no_grad():
        all_predictions = []
        # 添加推理进度条
        for batch in tqdm(predict_dataloder, desc="Inference Progress", total=len(predict_dataloder)):
            imgs, masks = batch

            last_cls, last_bbox = predict_model(imgs.to(device), masks.to(device), return_all_layers=False)

            pred_cls_probability = F.softmax(last_cls, dim=-1)  # shape: (batch_size, 100, num_classes+1)
            pred_bbox = xywh_to_xyxy(last_bbox)  # shape: (batch_size, 100, 4)

            batch_preds = []
            for i in range(pred_cls_probability.size(0)):
                each_img_cls_pre = pred_cls_probability[i]  # shape: (100, num_classes+1)
                max_prob_of_query, max_prob_of_query_cls_idx = torch.max(each_img_cls_pre, dim=-1)

                # 筛选概率大于 conf_thres 且类别不是 background_cls=num_classes 的预测框
                high_conf_mask = (max_prob_of_query >= conf_thres) & (max_prob_of_query_cls_idx != num_classes)
                
                pred_cls = max_prob_of_query_cls_idx[high_conf_mask] # 筛选得到的类别, dim = (n, )
                pred_cls_score = max_prob_of_query[high_conf_mask] # 筛选的类别的概率，也就是置信度分数, dim = (n, )
                pred_boxes = pred_bbox[i][high_conf_mask] # 筛选得到的bbox, dim = (n, 4)
                
                if NMS[0] == True:
                    iou_threshold = NMS[1]
                    # NMS非极大值抑制
                    # DETR虽然不需要NMS, 但是在训练不充足的时候, 仍然会有重叠框, 但是由于检测框数量<100, nms的成本很低, 在训练不足的情况下NMS抑制会比较好
                    # boxes: 边界框坐标 [x1, y1, x2, y2] 格式，形状为 [N, 4] 的Tensor, 同一个类别
                    # scores: 每个框的置信度，形状为 [N] 的Tensor, 同一个类别
                    # iou_threshold: IoU阈值
                    filtered_boxes_list = []
                    filtered_cls_list = []
                    filtered_score_list = []

                    # 获取所有存在的类别
                    unique_classes = torch.unique(pred_cls)

                    # 对每个类别单独进行NMS
                    for cls in unique_classes:
                        # 创建当前类别的掩码
                        cls_mask = (pred_cls == cls)
                        
                        # 获取当前类别的框、分数和类别
                        cls_boxes = pred_boxes[cls_mask]
                        cls_scores = pred_cls_score[cls_mask]
                        cls_labels = pred_cls[cls_mask]  # 实际都是同一个类别值
                        
                        # 如果当前类别没有框，跳过
                        if cls_boxes.numel() == 0:
                            continue
                        
                        # 对当前类别应用NMS
                        keep = nms(cls_boxes, cls_scores, iou_threshold)
                        
                        # 添加到结果列表
                        filtered_boxes_list.append(cls_boxes[keep])
                        filtered_cls_list.append(cls_labels[keep])
                        filtered_score_list.append(cls_scores[keep])

                    # 如果存在有效检测结果，合并所有类别
                    if filtered_boxes_list:
                        filtered_boxes = torch.cat(filtered_boxes_list)
                        filtered_cls = torch.cat(filtered_cls_list)
                        filtered_cls_score = torch.cat(filtered_score_list)
                    else:
                        # 如果没有检测结果，创建空张量
                        filtered_boxes = torch.empty((0, 4), device=pred_boxes.device)
                        filtered_cls = torch.empty((0,), dtype=torch.long, device=pred_cls.device)
                        filtered_cls_score = torch.empty((0,), device=pred_cls_score.device)
                    
                    # 每一张图片的预测情况，写入batch_preds
                    batch_preds.append((filtered_boxes, filtered_cls, filtered_cls_score))
                else:
                    batch_preds.append((pred_boxes, pred_cls, pred_cls_score))
            # 整个批次的图片的预测情况
            all_predictions.append(batch_preds)

        # 写入id, xyxy, score内容到pred_dict字典内
        num = 0
        pred_dict = {}
        img_name_lst = os.listdir(imgdir)
        # 添加归一化坐标写入内置字典进度条
        pbar = tqdm(total=len(img_name_lst), desc="write into inner dict")
        # 批次循环
        for batch_preds in all_predictions:
            # 批内循环
            for pred in batch_preds:
                pred_boxes, pred_cls, highest_score = pred

                imgname_str = img_name_lst[num]
                index = imgname_str.find(".")
                txt_name = imgname_str[ : index] + ".txt"

                if len(pred_boxes) > 0:  # 只有当有预测框时才写入(也就是检出框个数大于0)
                    pred_txt_name = txt_name
                    img_pred_idxyxy = []
                    # 图内检测框循环
                    for cls_id, bbox, score in zip(pred_cls, pred_boxes, highest_score):
                        line = [cls_id.item(), round(bbox[0].item(), 6), round(bbox[1].item(),6), 
                                round(bbox[2].item(), 2), round(bbox[3].item(),2), round(score.item(), 2)]
                        img_pred_idxyxy.append(line)
                    pred_dict[pred_txt_name] = img_pred_idxyxy

                # 更新进度条
                pbar.update(1)
                num += 1
        pbar.close()

    # 反归一化为预测图像上的检测框坐标
    un_normalized_dict = {}
    pbar = tqdm(total=len(pred_dict), desc="coordination denormalize")
    for pred_txt_name, idxyxy_score in pred_dict.items():
        index = pred_txt_name.find(".")
        info = pred_txt_name[:index] # txt的前缀名

        # 动态查找图片
        possible_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in possible_extensions:
            test_path = os.path.join(imgdir, info + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"Warning: not find pic {info + ext}, skipped!")
            continue
            
        # 打开图片并处理
        img = Image.open(img_path)
        w, h = img.size
        tw, th = imgsz
        scale = min(tw/w, th/h)
        new_w, new_h = scale*w, scale*h
        
        scale_w = tw/new_w
        scale_h = th/new_h

        each_img_real_box = []
        new_txt_lst = []
        for item in idxyxy_score:
            cls_id, xmin_n, ymin_n, xmax_n, ymax_n, score = item[0], item[1], item[2], item[3], item[4], item[5]
            Xmin_r = xmin_n * scale_w * w
            Ymin_r = ymin_n * scale_h * h
            Xmax_r = xmax_n * scale_w * w
            Ymax_r = ymax_n * scale_h * h
            
            real_box_line = [cls_id, round(Xmin_r, 6), round(Ymin_r, 6), round(Xmax_r, 6), round(Ymax_r, 6), score]
            each_img_real_box.append(real_box_line)
        
            
            str_line = f"{cls_id} {round(Xmin_r, 6)} {round(Ymin_r, 6)} {round(Xmax_r, 6)} {round(Ymax_r, 6)} {score}"
            new_txt_lst.append(str_line)
        
        if output_dir != None:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, info+".txt"), 'w') as f:
                for j in new_txt_lst:
                    f.write(j + '\n')
        
        un_normalized_dict[pred_txt_name] = each_img_real_box

        pbar.update(1)
    pbar.close

    return un_normalized_dict

# 生成n个类别的独特颜色（HSV空间均匀分布）
def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # 在HSV空间均匀分布色调
        hue = i * (180 / num_colors)  # OpenCV的H范围是0-180
        # 固定饱和度和亮度为高值，确保颜色鲜艳
        color_hsv = (hue, 255, 255)
        # 转换为BGR格式
        color_bgr = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors

# 获取目标目录中所有图片文件
def get_image_files(directory):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(file)
    return image_files

def predict_plot_fn(class_file, img_dir, dict_unormalize, predict_plot_dir):
    # 确保输出目录存在
    os.makedirs(predict_plot_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------
    # 颜色生成模块
    class_names = []
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # 为所有类别生成独特颜色
    num_classes = len(class_names)
    class_colors = generate_distinct_colors(num_classes)
    # ------------------------------------------------------------------------------------------------------

    # 获取所有目标图片文件
    target_images = get_image_files(img_dir)
    total_images = len(target_images)
    # print(f"共找到 {total_images} 张目标图片需要处理")

    # 统计变量
    processed_images = 0
    total_boxes = 0

    # 使用tqdm创建进度条
    for img_file in tqdm(target_images, desc="bounding box ploting"):
        # 获取图片文件名（不带扩展名）
        file_name, _ = os.path.splitext(img_file)
        
        # 加载目标图像
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)

        # 计算基础缩放因子（使用对角线长度作为基准）
        h, w = img.shape[:2]
        diag = math.sqrt(h**2 + w**2)
        base_scale = diag / 1200  # 1200 是经验值，可根据需求调整

        # 计算自适应矩形框线宽
        line_thickness = max(1, min(8, int(round(base_scale * 3))))
        
        # 检查是否有对应的预测值存储
        pred_txt_name = f"{file_name}.txt"
        
        if pred_txt_name in dict_unormalize:
            # 读取dict中的检测框信息
            pred_box = dict_unormalize[pred_txt_name]
            boxes_in_image = 0
            
            # 绘制检测框
            for box in pred_box:
                # 解析每一行的类别和位置信息
                if len(box) < 6:  # 确保有足够的数据
                    continue
        
                try:
                    class_id = int(box[0])  # 类别ID
                    xmin = int(float(box[1]))   # xmin坐标
                    ymin = int(float(box[2]))   # ymin坐标
                    xmax = int(float(box[3]))   # xmax坐标
                    ymax = int(float(box[4]))   # ymax坐标
                    score = float(box[5])
                
                    # 获取类别名称和颜色
                    class_name = class_names[class_id] if class_id < num_classes else f"Class_{class_id}"
                    color = class_colors[class_id] if class_id < num_classes else (0, 0, 0)  # 黑色作为默认
                    
                    # 绘制矩形框
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
                    
                    # 准备标签文本 (类别名称 + ID)
                    label_text = f"{class_name}: {score}"
                    
                    # 计算文本尺寸
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    
                    # 2. 动态计算线宽（最小1px，最大8px）
                    text_thickness = max(1, min(8, int(round(base_scale * 3))))
                    # 3. 动态计算字体大小（最小0.4，最大1.5）
                    font_scale = max(0.4, min(1.5, base_scale * 1.2))
                    
                    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                    
                    # 创建文本背景矩形
                    bg_rect_top_left = (xmin, ymin - text_height - 5)
                    bg_rect_bottom_right = (xmin + text_width, ymin)
                    
                    # 绘制文本背景
                    cv2.rectangle(img, bg_rect_top_left, bg_rect_bottom_right, color, -1)  # 填充矩形
                    cv2.rectangle(img, bg_rect_top_left, bg_rect_bottom_right, color, 2)  # 边框
                    
                    # 绘制文本
                    text_origin = (xmin, ymin - 5)
                    cv2.putText(img, label_text, text_origin, font, font_scale, (255, 255, 255), text_thickness)
                    
                    boxes_in_image += 1
                except Exception as e:
                    print(f"处理图片 {img_file} 时出错: {str(e)}")
                    continue
            
            total_boxes += boxes_in_image
        else:
            # 没有预测文件，只保存原图
            boxes_in_image = 0
        
        # 保存图像
        predict_path = os.path.join(predict_plot_dir, f"{file_name}.jpg")
        try: 
            cv2.imwrite(predict_path, img)
            processed_images += 1
        except Exception as e:
            print(f"保存图片 {img_file} 时出错: {str(e)}")

    # 输出统计信息
    print("\nFinished! Statistic info:")
    print(f"Total Photos: {total_images}")
    print(f"Pred Photos: {len(dict_unormalize)}")
    print(f"Processed Photos: {processed_images}")
    print(f"Plotting {total_boxes} bounding boxes")
    print(f"Using {num_classes} classes")
    return None
    
if __name__ =="__main__":
    # 文件读取区
    class_file = 'your_classes_file_path/classes_coco.txt'
    test_imgdir = "your_predict_img_path"
    predict_pic_dir = "your_path_to_save_predict_img"
    output_real_labels = "your_path_to_save_predict_real_labels_on_you_imgs"
    model_path = "your_path_to_model_weights/coco_weights/model_best_coco640.pth"
    # model_path = "your_path_to_model_weights/coco_weights/model_best_coco320.pth"

    # 模型参数配置
    num_classes = 80
    num_queries = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NMS操作启用就用True, 不用就False, output_dir保存你的预测框坐标数据, None就不保存
    dict_unormalize_file = predict_fn(imgdir=test_imgdir, model_weights=model_path, imgsz=(640,640), conf_thres=0.7,
                                      num_classes=num_classes, num_queries=num_queries, num_encoder_layer=6, num_decoder_layer=6,
                                      num_workers=0, pin_memory=True, device=device, batch_size=4, NMS=(True, 0.5), 
                                      output_dir=output_real_labels)
    
    # dict_unormalize_file = predict_fn(imgdir=test_imgdir, model_weights=model_path, imgsz=(640,640), conf_thres=0.7,
    #                                   num_classes=num_classes, num_queries=num_queries, num_encoder_layer=6, num_decoder_layer=6,
    #                                   num_workers=0, pin_memory=True, device=device, batch_size=4, NMS=(True, 0.5), 
    #                                   output_dir=None)
        
    predict_plot_fn(class_file=class_file, img_dir=test_imgdir, dict_unormalize=dict_unormalize_file, predict_plot_dir=predict_pic_dir)
