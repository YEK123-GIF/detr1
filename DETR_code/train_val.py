# 本文件不使用yaml文件配置方式, 通过修改代码进行指定路径和配置

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from DetectionTransformer import DETR_model
from dataset_process_train import TrainDataset_for_DETR
from loss_compute_aux import compute_loss
from utils import xywh_to_xyxy, collate_fn_train_val, caculate_num_queries
from validator import validate_model, printer_val

# 经验公式: average_gt*1 = (num_queries-average_gt)*background_weight
# 两种调参法: 降低queries数量, 或者降低背景权重, 建议降低queries数量, 加速收敛
# ============================= 主训练函数 =============================
if __name__ == '__main__':
    # 路径配置
    train_imgdir_path = "C:/Users/29459/OneDrive/Desktop/Silksong/images/train"
    train_txtdir_path = "C:/Users/29459/OneDrive/Desktop/Silksong/labels/train"
    
    val_imgdir_path = "C:/Users/29459/OneDrive/Desktop/Silksong/images/val"
    val_txtdir_path = "C:/Users/29459/OneDrive/Desktop/Silksong/labels/val"
    
    model_best_path = "E:/PythonProject3/detr/weights/model_best_coco.pth"
    model_last_path = "E:/PythonProject3/detr/weights/model_last_coco.pth"

    # 训练配置
    num_classes = 80
    num_queries = 100
    
    proposal_num_queries = caculate_num_queries(txtdir=train_txtdir_path, background_weight=0.1)
    print(f"自适应查询框推荐数量: {proposal_num_queries}")

    imgsz = (640, 640)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    
    epochs = 500
    learning_rate = 1e-4
    weight_decay = 1e-4
    scores_threshold = 0.001  # 验证时的置信度阈值

    # 初始化模型
    my_model = DETR_model(num_queries=num_queries, num_classes=num_classes).to(device)
    trainer = AdamW(params=my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 创建数据集
    train_dataset = TrainDataset_for_DETR(train_imgdir_path, train_txtdir_path, target_size=imgsz)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train_val)

    # 初始化最佳指标
    best_map = 0.0
    best_map50 = 0.0
    best_map50t95 = 0.0

    for epoch in range(epochs):
        # ===================== 训练阶段 =====================
        my_model.train()
        epoch_losses = {'total': 0.0, 'cls': 0.0, 'l1': 0.0, 'giou': 0.0}
        batch_num = 0
        
        t = tqdm(train_loader, unit='batch', desc=f'训练 Epoch {epoch+1}/{epochs}')
        for batch in t:
            img_info, masks, real_id, real_bbox, _ = batch

            real_bbox = xywh_to_xyxy(real_bbox) #用xyxy计算匈牙利与loss
            
            all_cls, all_bbox = my_model(img_info.to(device), mask=masks.to(device))
            
            total_loss, loss_dict = compute_loss(
                all_cls, 
                all_bbox, 
                real_id, 
                real_bbox, 
                device=device,
                background_weight=0.1
            )
            
            total_loss.backward()
            # 梯度裁剪, 使得范数<=max_norm=0.1, 范数就是所有参数梯度的绝对值之和(L1), 或者参数梯度平方和开根(L2)
            # 默认使用L2范数, 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=0.1)
            trainer.step()
            trainer.zero_grad()

            batch_num += 1
            epoch_losses['total'] += total_loss.item()
            epoch_losses['cls'] += loss_dict['cls_loss']
            epoch_losses['l1'] += loss_dict['l1_loss']
            epoch_losses['giou'] += loss_dict['giou_loss']

            t.set_postfix({
                'cls_loss': loss_dict['cls_loss'],
                'l1_loss': loss_dict['l1_loss'],
                'giou_loss': loss_dict['giou_loss'],
                'total_loss': total_loss.item()
            })
        
        # 打印训练统计
        avg_total = epoch_losses['total'] / batch_num
        avg_cls = epoch_losses['cls'] / batch_num
        avg_l1 = epoch_losses['l1'] / batch_num
        avg_giou = epoch_losses['giou'] / batch_num
        
        print(f"Epoch {epoch+1} 训练总结:")
        print(f"  cls_loss: {avg_cls:.4f}  l1_loss: {avg_l1:.4f}  giou_loss: {avg_giou:.4f}")
        print(f"  总损失: {avg_total:.4f}")

        # ===================== 验证阶段 =====================
        print(f"\n开始验证 Epoch {epoch+1}...")
        val_metrics, all_p50, all_r50, current_map50, current_map75, current_map50t95 = validate_model(
            val_imgdir_path, val_txtdir_path, model=my_model, 
            num_classes=num_classes, num_queries=num_queries, 
            imgsz=imgsz, batch_size=4, workers=0, pin_memory=True, 
            scores_threshold=scores_threshold
        ) #用于验证模型
        # 验证部分打印: 
        printer_val(val_metrics, all_p50, all_r50, current_map50, current_map75, current_map50t95)
        
        current_map = 0.1 * current_map50 + 0.9 * current_map50t95
        # 保存最佳模型
        if current_map > best_map:
            best_map = current_map
            best_map50 = current_map50
            best_map50t95 = current_map50t95
            
            torch.save(my_model.state_dict(), model_best_path)
            print(f"保存最佳模型到: {model_best_path}")
        
        # 每个epoch都保存最新模型
        torch.save(my_model.state_dict(), model_last_path)
        print(f"保存最新模型到: {model_last_path}\n")

    print(f"训练完成! 最佳 mAP@50: {best_map50:.4f}, 最佳mAP@50~95: {best_map50t95}")

    print(f"best weight summary: ")
    best_val_metrics, best_all_p50, best_all_r50, best_map50, best_map75, best_map50t95 = validate_model(
                                        val_imgdir_path, val_txtdir_path, model_path=model_best_path, 
                                        num_classes=num_classes, num_queries=num_queries, 
                                        imgsz=imgsz, batch_size=batch_size, workers=0, pin_memory=True, 
                                        scores_threshold=scores_threshold
                                    )
    # 验证部分打印: 
    printer_val(best_val_metrics, best_all_p50, best_all_r50, best_map50, best_map75, best_map50t95)