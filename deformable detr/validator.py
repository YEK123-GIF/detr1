import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DetectionTransformer import DETR_model
from dataset_process_train import TrainDataset_for_DETR
from utils import xywh_to_xyxy, collate_fn_train_val

def calculate_iou(box1, box2):
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])
    
    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

def convert_yolo_to_xyxy(bbox):
    """用于转换后计算IOU"""
    x_center, y_center, width, height = bbox
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    return [xmin, ymin, xmax, ymax]

def calculate_metrics(gt_data:dict, pred_data:dict, iou_threshold=0.5):
    # 统计每个类别的全局真实框数量
    class_gt_counts = defaultdict(int)
    for _, gt_image in gt_data.items():
        for class_id, bboxes in gt_image.items():
            class_gt_counts[class_id] += len(bboxes)

    # 统计验证集的所有类别
    all_gt_classes = set(class_gt_counts.keys())

    class_results = defaultdict(list)
    all_img_names = set(gt_data.keys())

    for image_id in all_img_names:
        gt_image:dict
        pred_image:dict
        gt_image = gt_data[image_id]
        pred_image = pred_data.get(image_id, {})
        
        # 遍历验证集所有类别
        for class_id in all_gt_classes:
            gt_boxes:dict
            pred_boxes:dict
            
            # 情况1：该图片有该类别的真实框
            if class_id in gt_image:
                gt_boxes = [convert_yolo_to_xyxy(bbox) for bbox in gt_image[class_id]]
                pred_boxes = pred_image.get(class_id, [])
                # x[1] = score, pred_boxes = [([x,y,x,y], score), ([x,y,x,y], score), ([x,y,x,y], score)....]
                # 按置信度排序, 高置信度的优先匹配, 只要通过IoU阈值就被认为是TP, 即使低置信度的IoU更高也被认为是FP。
                pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
                
                gt_matched = [False] * len(gt_boxes)
                
                # (1) pred_boxes_sorted = [], for循环不执行, 该真实类别的预测框不存在, 或者出现验证集类别以外的类别
                # (2) pred_boxes_sorted != [], for循环执行, 判断预测框是tp, fp
                for pred_box, score in pred_boxes_sorted:
                    best_iou = 0.0
                    best_idx = -1

                    # this one pred_box matches all gt_boxes one by one
                    for i, gt_box in enumerate(gt_boxes):
                        if gt_matched[i]:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    
                    # (1) best_iou = 0 or best_iou < iou_threshold, is_tp = False
                    # (2) best_iou >= iou_threshold, is_tp = True
                    is_tp = best_iou >= iou_threshold
                    class_results[class_id].append({
                        'score': score,
                        'tp': is_tp,
                    })
                    
                    if is_tp and best_idx >= 0:
                        gt_matched[best_idx] = True
            
            # 情况2：该图片没有该类别的真实框
            else:
                pred_boxes = pred_image.get(class_id, [])
                # (1) pred_boxes = [], for循环不执行, 该真实类别的预测框不存在, 或者出现验证集类别以外的类别
                # (2) pred_boxes != [], 所有预测框都是误检fp
                for pred_box, score in pred_boxes:
                    class_results[class_id].append({
                        'score': score,
                        'tp': False  # 明确标记为FP
                    })
        # 以上代码处理逻辑
        # (1) 该类别下: pred_box, gt_box均存在, 正常计算tp, fp
        # (2) 该类别下: pred_box=[], gt_box存在, 漏检, 通过total_gt计算fn
        # (3) 该类别下: pred_box存在, gt_box=[], 误检, 直接标记为fp
        # (4) 该类别下: pred_box=[], gt_box=[], 说明检测正常无需担忧(不做标记)
    
    gt_class_metrics = {}
    gt_class_ap = [] # 仅存储验证集类别的AP, 方便用于计算验证集类别下的ap
    all_classes_results = [] # all_classes_results用于存储所有验证集类别结果, 方便进行全局tp, fp统计

    # 对于验证集中所存在的类别的真实框计算mAP所需, 计算每一个类别的mAP值
    for class_id in all_gt_classes:
        # 用真实gt框类别, 映射预测框类别
        results = class_results[class_id]
        # x={'score':..., 'tp':...}字典, result=[{},{},{}...]是一个可迭代列表
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # 使用全局真实框数量(tp+fn, 仅验证集中存在的类别)
        class_gt = class_gt_counts[class_id] # 总是>0
        
        tp_cum, fp_cum = 0, 0
        precisions, recalls, confs = [], [], []

        # 求解所有预测框计算出来的P、R值---->制作(P, R)散点图
        # score越大, 过滤掉的pred越多, P分母越小, tp越大, 最终达到1, R的分母是总真实框, 当pred为1, R会很低 
        # results_sorted = [] 不运行for循环
        for result in results_sorted:
            # if True: tp就计数+1
            if result['tp']:
                tp_cum = tp_cum + 1
            # if False: fp就计数+1
            else:
                fp_cum = fp_cum + 1
            # 获取每一个置信度下的p, r, conf(score)
            precision_i = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
            recall_i = tp_cum / class_gt if class_gt > 0 else 0
            conf_i = result["score"] # 获取此预测框的置信度score

            precisions.append(precision_i)
            recalls.append(recall_i)
            confs.append(conf_i)
        all_classes_results = all_classes_results + results
        
        # iou_thres=iou下, 根据每一个真实存在的类别的(P, R), 依次计算F1值, 当F1最大值时, 返回(P, R), 以及其所对应的score值(conf值)
        f1_lst = []
        for i in range(len(confs)):
            if precisions[i] + recalls[i] == 0:
                f1 = 0
                f1_lst.append(f1)
            else:
                f1 = 2*precisions[i]*recalls[i]/(precisions[i]+recalls[i])
                f1_lst.append(f1)
        
        # 处理验证集中某类别没有预测框, 也就是 pred_bsx = [], 导致results = [], 这一类的p, r, score 都为 0
        if f1_lst != [] :
            f1_max = max(f1_lst)
            index_max_f1 = f1_lst.index(f1_max)

            f1_max_precision = precisions[index_max_f1]
            f1_max_recall = recalls[index_max_f1]
            f1_max_score = confs[index_max_f1]
        else:
            f1_max_precision = 0
            f1_max_recall = 0
            f1_max_score = 0

        # ap值计算, 使用101点插值法
        ap = 0.0
        if recalls and precisions:
            interp_precisions = []
            for t in np.arange(0, 1.01, 0.01):
                prec_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
                if prec_at_recall:
                    interp_precisions.append(max(prec_at_recall))
                else:
                    interp_precisions.append(0)
            ap = np.mean(interp_precisions)
            gt_class_ap.append(ap)
        
        tp_total = sum(1 for r in results_sorted if r['tp'])
        fp_total = sum(1 for r in results_sorted if not r['tp'])

        # 构建每一个类别的p, r, ap值, 以及对应的tp, fp, gt, 其中p, r是根据最大f1动态得出的, ap是所有置信度下的积分
        gt_class_metrics[class_id] = {
            'precision': f1_max_precision,
            'recall': f1_max_recall,
            'precision_recall_score': f1_max_score,
            'ap': ap,
            'tp': tp_total,
            'fp': fp_total,
            'gt': class_gt,
        }
    
# =============================================================================================================
# 求解全体class下的P, R计算出的F1=2*p*r/(p+r), F1最大值所对应的(P, R)值与其对应的confs
# =============================================================================================================
    all_classes_results_sorted = sorted(all_classes_results, key=lambda x: x['score'], reverse=True)

    total_gt = sum([class_gt_counts[cls_gt] for cls_gt in all_gt_classes])
    all_tp_cum, all_fp_cum = 0, 0
    all_precisions, all_recalls, all_confs = [], [], []
    # 求解全局预测框计算出来的P、R值, score越大, 过滤掉的pred越多, P分母越小, tp越大, 最终达到1, R的分母是总真实框, 当pred为1, R会很低 
    for all_class_result in all_classes_results_sorted:
        # if True: tp就计数+1
        if all_class_result['tp']:
            all_tp_cum = all_tp_cum + 1
        # if False: fp就计数+1
        else:
            all_fp_cum = all_fp_cum + 1
        # 获取每一个置信度下的p, r, conf(score)
        all_precision_i = all_tp_cum / (all_tp_cum + all_fp_cum) if (all_tp_cum + all_fp_cum) > 0 else 0
        all_recall_i = all_tp_cum / total_gt if total_gt > 0 else 0
        all_conf_i = all_class_result["score"] # 获取此预测框的置信度score

        all_precisions.append(all_precision_i)
        all_recalls.append(all_recall_i)
        all_confs.append(all_conf_i)
    
    # iou_thres=iou下, 根据全体数据集的(P, R), 依次计算F1值, 当F1最大值时, 返回(P, R), 以及其所对应的score值(conf值)
    all_f1_lst =[]
    for j in range(len(all_confs)):
        if all_precisions[j] + all_recalls[j] == 0:
            all_f1 = 0
            all_f1_lst.append(all_f1)
        else:
            all_f1 = 2*all_precisions[j]*all_recalls[j]/(all_precisions[j]+all_recalls[j])
            all_f1_lst.append(all_f1)
    # 处理某真实类别没有预测框的情况
    if all_f1_lst != []:
        all_f1_max = max(all_f1_lst)
        all_index_max_f1 = all_f1_lst.index(all_f1_max)

        all_f1_max_precision = all_precisions[all_index_max_f1]
        all_f1_max_recall = all_recalls[all_index_max_f1]
        all_f1_max_score = all_confs[all_index_max_f1]
    else:
        all_f1_max_precision = 0
        all_f1_max_recall = 0
        all_f1_max_score = 0
    
    # 求解map_iou值, 这是正确求解方法, 而不是用全局的P, R曲线计算全局AP作为mAP
    # 因为每个类别同等重要, 如果用全局PR曲线, 会导致优势类别主导, 分数较高, 没有意义
    map_iou = np.mean(gt_class_ap) if gt_class_ap else 0
    
    return gt_class_metrics, all_f1_max_precision, all_f1_max_recall, all_f1_max_score, map_iou

def validate_model(val_imgpath, val_txtpath, model_path=None, model=None, num_classes=80, num_queries=100, 
                   imgsz=(800, 800), batch_size=4, workers=0, pin_memory=True, scores_threshold=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建数据集和数据加载器
    my_dataset = TrainDataset_for_DETR(val_imgpath, val_txtpath, target_size=imgsz)
    mydataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=collate_fn_train_val, num_workers=workers, pin_memory=pin_memory)

    # 加载模型
    if model_path != None and model == None:
        state_dict = torch.load(model_path, map_location=device)
        my_model = DETR_model(num_queries=num_queries, num_classes=num_classes, 
                            num_encoder_layer=6, num_decoder_layer=6).to(device)
        my_model.load_state_dict(state_dict)
        my_model.eval()
    elif model_path == None and model != None:
        my_model = model
        my_model:DETR_model
        my_model.eval()
    else:
        raise ValueError('Running Error!')

    # 存储真实标注和预测结果
    gt_data_all = defaultdict(lambda: defaultdict(list))
    pred_data_all = defaultdict(lambda: defaultdict(list))
    
    img_name_lst = os.listdir(val_imgpath)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(mydataloader, desc="Verification Progress")):
            imgs, masks, _, _, w1h1 = batch  # 忽略不需要的real_id, real_bbox
            
            # 模型推理
            last_cls, last_bbox = my_model(imgs.to(device), masks.to(device), return_all_layers=False)
            pred_cls_probability = F.softmax(last_cls, dim=-1)
            pred_bbox = xywh_to_xyxy(last_bbox)
            
            # 处理批次中的每个图像
            for i in range(imgs.size(0)):
                img_idx = batch_idx * mydataloader.batch_size + i
                img_name = img_name_lst[img_idx]
                img_id = os.path.splitext(img_name)[0]
                
                # 处理真实标注
                txt_path = os.path.join(val_txtpath, f"{img_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            class_id = int(parts[0])
                            bbox = list(map(float, parts[1:5]))
                            
                            # 转换真实标注到变换后的图像尺寸w,h ---> new_w, new_h,也就是w1,h1
                            w1, h1 = w1h1[i][0], w1h1[i][1]
                            tw, th = imgsz
                            x_center = bbox[0] * w1 / tw
                            y_center = bbox[1] * h1 / th
                            width = bbox[2] * w1 / tw
                            height = bbox[3] * h1 / th
                            # gt_data_all: {img_id0: {class_id0: [[x_center, y_center, width, height], [x_center, y_center, width, height]],
                            #                        class_id1: [[x_center, y_center, width, height]]},
                            #               img_id1: {class_id0: [[x_center, y_center, width, height]], 
                            #                        class_id2: [[x_center, y_center, width, height]]} }
                            gt_data_all[img_id][class_id].append([x_center, y_center, width, height])
                
                # 处理预测结果
                each_img_cls_pre = pred_cls_probability[i]
                max_prob, max_cls_idx = torch.max(each_img_cls_pre, dim=-1)
                
                high_conf_mask = (max_prob >= scores_threshold) & (max_cls_idx != num_classes)
                pred_cls_high_conf = max_cls_idx[high_conf_mask] # (N,)
                pred_cls_score = max_prob[high_conf_mask] # (N,)
                pred_boxes_high_conf = pred_bbox[i][high_conf_mask] # (N, 4)
                
                for cls_id, bbox, score in zip(pred_cls_high_conf, pred_boxes_high_conf, pred_cls_score):
                    # pred_data_all: {img_id0: {class_id0: [([x_center, y_center, width, height], score), 
                    #                                       ([x_center, y_center, width, height], score],
                    #                           class_id1: [ ([x_center, y_center, width, height], score) ]},
                    #                 img_id1: {class_id0: [([x_center, y_center, width, height], score)], 
                    #                          class_id2: [([x_center, y_center, width, height], score)]} }                    
                    pred_data_all[img_id][cls_id.item()].append((bbox.tolist(), score.item()))
    
    # 计算评估指标
    iou_list = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    class_metrics_list = []
    map_list = []
    all_class_p = []
    all_class_r = []
    all_class_score = []
    
    for iou in iou_list:
        gt_class_metrics, all_p_iou, all_r_iou, all_score_iou, map_iou = calculate_metrics(gt_data_all, pred_data_all, iou_threshold=iou)
        
        class_metrics_list.append(gt_class_metrics)
        all_class_p.append(all_p_iou)
        all_class_r.append(all_r_iou)
        all_class_score.append(all_score_iou)
        map_list.append(map_iou)
        
    map50 = map_list[0]
    map75 = map_list[5]
    map50t95 = np.mean(np.array(map_list))

    class_metrics50 = class_metrics_list[0]
    class_metrics50:dict
    id_lst = list(class_metrics50.keys())

    all_class_p50 = all_class_p[0]
    all_class_r50 = all_class_r[0]

    # 统计该类别的真实框数量, 是一个列表, total_gt = sum(gt_num), gt_num中每一个元素是对应类别gt个数
    gt_num = [class_metrics50[idx]['gt'] for idx in id_lst]
# ======================================================================
# 获取不同iou阈值下各个类别的计算结果
# ======================================================================
    p_iou, r_iou, ap_iou = [], [], []
    for metrics in class_metrics_list:
        # 用于存储当前iou阈值下metrics指标下每一个类别的结果
        idx_p, idx_r, idx_ap = [], [], []
        for idx in id_lst:
            class_idx_p = metrics[idx]['precision']
            class_idx_r = metrics[idx]['recall']
            class_idx_ap = metrics[idx]['ap']
            idx_p.append(class_idx_p)
            idx_r.append(class_idx_r)
            idx_ap.append(class_idx_ap)
        p_iou.append(idx_p)
        r_iou.append(idx_r)
        ap_iou.append(idx_ap)
    
    # 每一行是不同iou, 每一列是不同idx类别, 按列内元素求和取平均就是该类别在50~95上的P50~95, R50~95, AP50~95
    p_iou = np.array(p_iou)
    r_iou = np.array(r_iou)
    ap_iou = np.array(ap_iou)

    p50 = p_iou[0, :] # 一行
    r50 = r_iou[0, :] # 一行
    ap50 = ap_iou[0, :] # 一行
 
    p50t95 = np.mean(p_iou, axis=0) # 一行
    r50t95 = np.mean(r_iou, axis=0) # 一行
    ap50t95 = np.mean(ap_iou, axis=0) # 一行

    val_metrics = {'id':id_lst, 'p50':p50, 'r50': r50, 'ap50': ap50, 'p50t95': p50t95, 'r50t95': r50t95, 'ap50t95': ap50t95, 'gt': gt_num}

    return val_metrics, all_class_p50, all_class_r50, map50, map75, map50t95

# 打印结果
def printer_val(val_metrics, all_p50, all_r50, map50, map75, map50t95):
    class_id = val_metrics['id']
    class_gt = val_metrics['gt']
    all_gt = sum(val_metrics['gt'])
    class_num = len(class_id)
    print("Class\tGroundTruth\tPrecision@Iou=0.5\tRecall@IoU=0.5\t\tmAP@IoU=0.5\t\tmAP@IoU=0.5~0.95")
    print(f"All\t{all_gt}\t\t{all_p50:.4f}\t\t\t{all_r50:.4f}\t\t\t{map50:.4f}\t\t\t{map50t95:.4f}")
    for i in range(class_num):
        print(f"{class_id[i]}\t{class_gt[i]}\t\t{val_metrics['p50'][i]:.4f}\t\t\t"
      f"{val_metrics['r50'][i]:.4f}\t\t\t{val_metrics['ap50'][i]:.4f}\t\t\t"
      f"{val_metrics['ap50t95'][i]:.4f}")
    print(f"\nValidation Set mAP Summary:\nmAP@50: {map50:.4f}\nmAP@75: {map75:.4f}\nmAP@50~95: {map50t95:.4f}\n")
    return None

if __name__ == '__main__':
    # 配置参数
    num_classes = 80
    num_queries = 100
    # 指定图像大小 (target_w, target_h)
    imgsz = (640, 640)
    # 验证集验证时使用的置信度, 不要改
    scores_threshold = 0.001

    # 验证集路径
    imgdir_path = "yor_path/val_img"
    txtdir_path = "/yor_pathval_txt"
    model_path = "yor_path/coco_weights/model_best_coco640.pth"

    val_metrics, all_p50, all_r50, map50, map75, map50t95 = validate_model(imgdir_path, txtdir_path, model_path, num_classes=num_classes, 
                                                        num_queries=num_queries, imgsz=imgsz, batch_size=4, workers=0, 
                                                        pin_memory=True, scores_threshold=scores_threshold)
    
    t = printer_val(val_metrics, all_p50, all_r50, map50, map75, map50t95)
