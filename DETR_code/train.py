# 本文件适合使用yaml配置文件配置自己的参数, 在终端执行训练
# yaml, argparse是常用的文件配置与指令输入形式, 可以实现配置文件与实际代码分离, 
# 用户在不修改代码的情况下, 通过指令接口和配置文件就可以实现修改参数, 简化使用流程
import yaml
import argparse

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from DetectionTransformer import DETR_model
from dataset_process_train import TrainDataset_for_DETR
from loss_compute_aux import compute_loss
from utils import xywh_to_xyxy, collate_fn_train_val, caculate_num_queries
from validator import validate_model, printer_val

# python trainer.py --config config.yaml --epochs 300 
# --imgsz 640,640 or --imgsz "640, 640", --imgsz "640,640"...

def parse_args():
    parser = argparse.ArgumentParser(description="DETR Training with YAML Config")
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--imgsz", type=str, help="Override image size")
    parser.add_argument("--num_workers", "-n_w", type=int, help="Override number of workers")
    parser.add_argument("--num_classes", "-n_c", type=int, help="Override number of classes")
    parser.add_argument("--num_queries", "-n_q", type=int, help="Override number of queries")
    parser.add_argument("--pin_memory", "-pm", type=bool, default=True, help="Override pin_memory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Force device selection")
    return parser.parse_args()

def load_config(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    
    if args.imgsz:
        w, h = map(int, args.imgsz.split(','))
        config['training']['imgsz'] = (w, h)

    if args.num_workers: config['training']['num_workers'] = args.num_workers
    if args.num_classes: config['training']['num_classes'] = args.num_classes
    if args.num_queries: config['training']['num_queries'] = args.num_queries
    if args.pin_memory: config['training']['pin_memory'] = args.pin_memory
    if args.device: config['training']['device'] = args.device
    return config

def main(config, device_override=None):
    # device choosing
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # path config
    paths = config['paths']
    train_config = config['training']
    
    # training config
    proposal_num_queries = caculate_num_queries(
        txtdir=paths['train_txtdir'],
        background_weight=train_config['background_weight']
    )
    print(f"Proposal number of queries: {proposal_num_queries}")
    print(f"Actually used num of queries: {train_config['num_queries']}\n")

    print("Training config and config path:")
    print("="*100)
    for k, v in paths.items():
        print(f"{k}: {v}")
    print("-"*100)
    for k1, v1 in train_config.items():
        print(f"{k1}: {v1}")
    print("="*100 + "\n")

    # model initialization
    my_model = DETR_model(
        num_queries=train_config['num_queries'],
        num_classes=train_config['num_classes']
    ).to(device)
    
    # optimizer
    trainer = AdamW(
        params=my_model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    # Training dataset
    train_dataset = TrainDataset_for_DETR(
        paths['train_imgdir'],
        paths['train_txtdir'],
        target_size=train_config['imgsz']
    )
    
    # Training dataset loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_train_val,
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory']
    )

    # training start
    best_map = 0.0
    best_map50 = 0.0
    best_map50t95 = 0.0
    
    for epoch in range(train_config['epochs']):
        my_model.train()
        epoch_losses = {'total': 0.0, 'cls': 0.0, 'l1': 0.0, 'giou': 0.0}
        batch_num = 0
        
        t = tqdm(train_loader, unit='batch', desc=f'Training Epoch {epoch+1}/{train_config["epochs"]}')
        for batch in t:
            img_info, masks, real_id, real_bbox, _ = batch
            real_bbox = xywh_to_xyxy(real_bbox)
            
            all_cls, all_bbox = my_model(img_info.to(device), mask=masks.to(device))
            
            total_loss, loss_dict = compute_loss(
                all_cls, all_bbox, real_id, real_bbox, 
                device=device,
                background_weight=train_config['background_weight']
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=0.1)
            trainer.step()
            trainer.zero_grad()

            batch_num += 1
            epoch_losses['total'] += total_loss.item()
            for k in ['cls', 'l1', 'giou']:
                epoch_losses[k] += loss_dict[f'{k}_loss']

            t.set_postfix({k: v for k, v in loss_dict.items()})
        
        # statistic of training epoch print
        print(f"\nEpoch {epoch+1} Training summary:")
        for k in ['cls', 'l1', 'giou', 'total']:
            avg = epoch_losses[k] / batch_num
            print(f"  {k}_loss: {avg:.4f}")

        # validate part
        print(f"\nValidating start: Epoch {epoch+1}...")
        val_metrics, all_p50, all_r50, current_map50, current_map75, current_map50t95 = validate_model(
            val_imgpath=paths['val_imgdir'],
            val_txtpath=paths['val_txtdir'],
            model=my_model,
            num_classes=train_config['num_classes'],
            num_queries=train_config['num_queries'],
            imgsz=train_config['imgsz'],
            batch_size=train_config['batch_size'],
            workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'],
            scores_threshold=train_config['scores_threshold']
        )
        
        printer_val(val_metrics, all_p50, all_r50, current_map50, current_map75, current_map50t95)
        
        # model save
        current_map = 0.1 * current_map50 + 0.9 * current_map50t95
        if current_map > best_map:
            best_map = current_map
            best_map50 = current_map50
            best_map50t95 = current_map50t95
            torch.save(my_model.state_dict(), paths['model_best'])
            print(f"Save best model to: {paths['model_best']}")
        
        torch.save(my_model.state_dict(), paths['model_last'])
        print(f"Save the latest model to: {paths['model_last']}\n")

    # summary of the train with validation
    print(f"Training Completed! Best mAP@50: {best_map50:.4f}, best mAP@50~95: {best_map50t95:.4f}")
    
    print("Validation result of the best model:")
    best_val_metrics, best_all_p50, best_all_r50, best_map50, best_map75, best_map50t95 = validate_model(
        paths['val_imgdir'], paths['val_txtdir'],
        model_path=paths['model_best'],
        num_classes=train_config['num_classes'],
        num_queries=train_config['num_queries'],
        imgsz=train_config['imgsz'],
        batch_size=train_config['batch_size'],
        workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        scores_threshold=train_config['scores_threshold']
    )
    # print the best model validation 
    printer_val(best_val_metrics, best_all_p50, best_all_r50, best_map50, best_map75, best_map50t95)

if __name__ == '__main__':
    # 创建解析器并添加参数
    args = parse_args()
    # 传入、覆盖、替换参数
    config = load_config(args.config, args)
    main(config, args.device)