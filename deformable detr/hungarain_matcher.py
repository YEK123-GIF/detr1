import torch
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from scipy.optimize import linear_sum_assignment

# 每一个txt包含多个框的框类别和框的4个坐标信息数据，读取后real_id一般是只有几个框
# real_id---->元组存储每张图的所有box对应的id: (tensor([1,0,1,2]), tensor([0,1,2])...)
# real_bbox----->元组存储每张图的所有box对应坐标：((a1, 4), (a2, 4)...), 其中a1~an: batch_size=n, ai代表的是一张图中的gt框数量

# 匈牙利匹配函数
def HungarianMatch(real_id:tuple, real_bbox:tuple, pred_id:torch.Tensor, pred_bbox:torch.Tensor):
    batch_size, num_queries = pred_id.shape[0:2]
    
    out_logits = F.softmax(pred_id, dim=-1) # (bs, 100, num_calsses+1)
    out_bbox = pred_bbox.flatten(0, 1) #(bs*100, 4)

    # real_id: 元组存储每张图的所有box对应的id: (tensor(1,0,1,2), tensor(0,1,2)...)
    # real_bbox:每张图的所有box对应的bbox: ((a1, 4), (a2, 4)...)
    target_id = torch.cat(real_id) # (num_id,)
    target_bbox = torch.cat(real_bbox, dim=0) # (num_bbox, bbox_xyxy=4)
    
    # class_cost
    out_pro = out_logits.view(-1, out_logits.size(-1))  # (bs*100, num_calsses+1)
    cls_cost = -out_pro[:, target_id] # (bs*100, all_true_id_in_the_batch's_num)

    # l1_cost, giou_cost
    l1_cost = torch.cdist(out_bbox, target_bbox, p=1) #torch.cdist(X, Y, p=1)会计算X，Y对应位置(i, j)的L1距离(p=1表示L1距离)并返回一个二维矩阵
    giou_cost = -box_ops.generalized_box_iou(out_bbox, target_bbox) #torchvision.ops.generalized_box_iou(A, B) 会两两计算 A 中每个框与 B 中每个框的 GIoU（Generalized IoU），返回一个矩阵，形状为 (len(A), len(B))

    C = cls_cost + 5 * l1_cost + 2 * giou_cost #通过权重来平衡不同cost的量纲的影响
    C = C.view(batch_size, num_queries, -1)

    # sizes是一个列表，里面每个元素代表每张图的gt框个数
    sizes = [len(i) for i in real_id]
    # C.split(sizes, -1)返回一个元组,为batch_size长度的元组
    indices = [linear_sum_assignment(c[i].detach().cpu().numpy()) for i, c in enumerate(C.split(sizes, -1))]
    # indices：输出batch_size个最优化元组每一个列表[(row_id1, col_id1), (row_id2, col_id2),...]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] #转换为tensor形式


# 以下三个用于独立计算cost, 有需要了解成本矩阵如何计算可以用, 不参与实际计算, 只做学习用
def Cls_cost(real_id, pred_id):
    # real_id: 元组存储每张图的所有box对应的id: (tensor(1,0,1,2), tensor(0,1,2)...)
    target = torch.cat(real_id) # (num_id,)
    # 预测值结果
    outputs = F.softmax(pred_id, dim=-1) # (bs, 100, num_calsses+1)
    out_pro = outputs.view(-1, outputs.size(-1))  # (32*100, num_calsses+1)
    cls_cost = -out_pro[:, target] # (32*100, all_true_id_in_the_batch's_num)
    return cls_cost # (batch_size*num_queries, num_of_true_box_id)

def L1_cost(real_bbox, pred_bbox):

    # real_bbox:每张图的所有box对应的bbox: ((a1, 4), (a2, 4)...)
    target_bbox = torch.cat(real_bbox, dim=0) # (num_bbox, bbox_xyxy=4)
    out_bbox = pred_bbox.flatten(0, 1) #(32*100, 4)
    l1_cost = torch.cdist(out_bbox, target_bbox, p=1)
    return l1_cost

def Giou_cost(real_bbox, pred_bbox):
    
    # real_bbox: 每张图的所有box对应的bbox: ((a1, 4), (a2, 4)...)
    target_bbox = torch.cat(real_bbox, dim=0) # (num_bbox, 4)
    out_bbox = pred_bbox.flatten(0, 1) #(32*100, 4)
    giou_cost = -box_ops.generalized_box_iou(out_bbox, target_bbox)
    return giou_cost



if __name__ =='__main__':
    real_id = (torch.tensor([0,1,1]),torch.tensor([0,1,1]))


    real_bbox = [torch.tensor( [[0.21, 0.35, 0.22, 0.45], 
                                [0.32, 0.14, 0.36, 0.26],
                                [0.12, 0.23, 0.21, 0.45]]),
                              
                 torch.tensor([[0.21, 0.35, 0.22, 0.45], 
                              [0.14, 0.14, 0.36, 0.26],
                              [0.40, 0.23, 0.51, 0.45]] )]
    
    output_bbox = torch.tensor([[[0.21, 0.35, 0.22, 0.45], [0.45, 0.23, 0.55, 0.41],[0.12, 0.23, 0.21, 0.45], 
                                [0.25, 0.36, 0.29, 0.65], [0.52, 0.62, 0.75, 0.98], [0.32, 0.14, 0.36, 0.26]],
                                
                                [[0.21, 0.35, 0.22, 0.45], [0.45, 0.23, 0.55, 0.41],[0.12, 0.23, 0.21, 0.45], 
                                [0.25, 0.36, 0.29, 0.65], [0.52, 0.62, 0.75, 0.98], [0.32, 0.14, 0.36, 0.26]]])
    
    output_id = torch.tensor([[[1.00, 0.10, 0.05, 0.05],[0.20, 0.30, 0.40, 0.10], [0.20, 4.68, 0.10, 0.10],
                              [0.15, 0.25, 0.25, 0.35],[0.10, 0.25, 0.30, 0.35], [0.20, 5.36, 0.05, 0.05]],
                              
                              [[1.00, 0.10, 0.05, 0.05],[0.20, 0.30, 0.40, 0.10], [0.20, 4.68, 0.10, 0.10],
                              [0.15, 0.25, 0.25, 0.35],[0.10, 0.25, 0.30, 0.35], [0.20, 5.36, 0.05, 0.05]]])
    

    print(f"output_bbox.shape:{output_bbox.shape}")
    print(f"output_id.shape:{output_id.shape}")

    cls_cost = Cls_cost(real_id, output_id)
    l1_cost = L1_cost(real_bbox, pred_bbox=output_bbox)
    giou_cost = Giou_cost(real_bbox=real_bbox, pred_bbox=output_bbox)
    matcher = HungarianMatch(real_id, real_bbox, output_id, output_bbox)
    print(f"匈牙利匹配结果:{matcher}")

    
