import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from ops.modules.ms_deform_attn import MSDeformAttn

class PatchEmbed(nn.Module):
    """提取特征并生成下采样后的掩码"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights='DEFAULT',  # 使用预训练权重
            trainable_layers=3  # 可训练层数（通常3或5）
        )

    def forward(self, x, mask=None):
        features = self.backbone(x)
        srcs = []
        spatial_shapes = []

        for name, feat in features.items():
            B, C, H, W = feat.shape
            srcs.append(feat.flatten(2).transpose(1, 2))  # [B, HW, C]
            spatial_shapes.append([H, W])

        src_flatten = torch.cat(srcs, dim=1)  # 拼接所有尺度 → [B, ΣHW, C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)
        # 处理掩码：使用最近邻插值进行下采样
        mask_pyramid = []
        if mask is not None:
            # 确保 mask 是四维张量 (bs, 1, H, W)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # 添加通道维度 (bs, 1, H, W)
            # 使用最近邻插值下采样到特征图尺寸
            mask_down = []
            for H, W in spatial_shapes:
                mask_down_i = F.interpolate(
                    mask.float(),
                    size=(H, W),
                    mode='nearest'
                )  # (bs, 1, H, W)
                mask_down_i = mask_down_i.squeeze(1) #(bs, H, W)
                mask_down_i = (mask_down_i > 0.5) #以0.5为判断阈值
                mask_pyramid.append(mask_down_i)
                mask_down_i = mask_down_i.flatten(1)# (bs, H*W)
                mask_down.append(mask_down_i)
            mask_down = torch.cat(mask_down, dim=-1)
            mask_down = mask_down.bool()  # True 表示填充位置，False 表示有效位置
        else:
            mask_down = None
            mask_pyramid = [None] * len(spatial_shapes)

        level_start_index = torch.cat((
            torch.zeros(1, dtype=torch.long, device=x.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))

        return src_flatten, spatial_shapes, level_start_index, mask_down, mask_pyramid
    
    
# encoder部分的二维可学习位置编码PositionEncode
class PositionEncode(nn.Module):
    def __init__(self, embed_dim=256, max_hw=800):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_hw = max_hw  # 最大高宽作为参数传进来

        # 根据最大高宽动态初始化 Embedding
        self.row_embed = nn.Embedding(max_hw, embed_dim // 2)
        self.col_embed = nn.Embedding(max_hw, embed_dim // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, h, w):
        assert h <= self.max_hw and w <= self.max_hw, f"输入尺寸超过最大支持尺寸 {self.max_hw}，请增大 max_hw"
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)   # (w, embed_dim//2)
        y_emb = self.row_embed(j)   # (h, embed_dim//2)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos.flatten(1, 2)  # (bs, h*w, embed_dim)

    def forward_hw(self, batch_size, h, w, device):
        dummy_x = torch.zeros(batch_size, 1, device=device)
        return self.forward(dummy_x, h, w)


class MultiScalePosEmbed(nn.Module):
    """多尺度位置编码 + 层级嵌入，输出与 src_flatten 对齐"""
    def __init__(self, embed_dim=256, num_levels=5, max_hw=512):
        super().__init__()
        self.pos_enc = PositionEncode(embed_dim=embed_dim, max_hw=max_hw)
        self.level_embed = nn.Embedding(num_levels, embed_dim)
        nn.init.normal_(self.level_embed.weight, std=0.02)

    @torch.no_grad()
    def _num_levels_from_spatial(self, spatial_shapes):
        # spatial_shapes: [L, 2]
        return spatial_shapes.shape[0]

    def forward(self, spatial_shapes, batch_size, device):
        """
        spatial_shapes: [L, 2] (H_l, W_l)，顺序需与 src_flatten 一致
        return:
          pos_flatten: [B, Σ(HW), C]  多尺度位置编码（含 level embedding）
        """
        L = spatial_shapes.shape[0]
        pos_list = []
        for l in range(L):
            H_l, W_l = spatial_shapes[l].tolist()
            pos_l = self.pos_enc.forward_hw(batch_size, H_l, W_l, device)  # [B, H_l*W_l, C]
            pos_l = pos_l + self.level_embed.weight[l].view(1, 1, -1)      # 加层级嵌入
            pos_list.append(pos_l)
        pos_flatten = torch.cat(pos_list, dim=1)                            # [B, Σ(HW), C]
        return pos_flatten

class DeformableDecoderQueries(nn.Module):
    """
    为 Deformable DETR decoder 提供：
      - content queries:      [B, Q, C]   (可学习，作为query的内容向量)
      - query pos embeddings: [B, Q, C]   (可学习，作为query的“位置编码”)
      - reference points:     [B, Q, L, 2] (每个query在每个尺度的参考点，归一化坐标)
    """
    def __init__(self, num_queries=300, embed_dim=256, num_feature_levels=4):
        super().__init__()
        # 可学习内容向量（原 DETR 里的 object queries）
        self.query_content = nn.Embedding(num_queries, embed_dim)
        # 可学习位置向量（与 content 分开，便于模块化）
        self.query_pos = nn.Embedding(num_queries, embed_dim)
        # 从位置向量预测 2D 参考点（[0,1] 范围）
        self.ref_point_head = nn.Linear(embed_dim, 2)
        self.num_feature_levels = num_feature_levels

        nn.init.normal_(self.query_content.weight, std=0.02)
        nn.init.normal_(self.query_pos.weight, std=0.02)
        nn.init.zeros_(self.ref_point_head.bias)
        nn.init.xavier_uniform_(self.ref_point_head.weight)

    def forward(self, batch_size, device=None):
        device = device if device is not None else self.query_content.weight.device
        # 内容向量 & 位置向量
        content = self.query_content.weight.unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # [B, Q, C]
        pos     = self.query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1).to(device)      # [B, Q, C]
        # 参考点（每个 query 一个 2D 点，后续广播到各层）
        ref_points_2d = self.ref_point_head(pos).sigmoid() #只需要一个参考中心点                                    # [B, Q, 2]
        # 扩展到多尺度： [B, Q, L, 2]
        reference_points = ref_points_2d.unsqueeze(2).repeat(1, 1, self.num_feature_levels, 1)
        return content, pos, reference_points

def get_reference_points(spatial_shapes, valid_ratios, device):
    """
    生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
    spatial_shapes: 4个特征图的shape [4, 2]
    valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
    device: cuda:0
    """
    reference_points_list = []
    # 遍历4个特征图的shape  比如 H_=100  W_=150
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
        # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
        # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5
        # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                      indexing='ij')
        # [100, 150] -> [bs, 15000]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        # [100, 150] -> [bs, 15000]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        # [bs, 15000, 2] 每一项都是xy
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
    # [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
    reference_points = torch.cat(reference_points_list, 1)
    # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
    # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
    # 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    # 4个flatten后特征图的归一化参考点坐标
    return reference_points

def get_valid_ratios_from_down_masks(mask_pyramid, spatial_shapes, device):
    ratios = []
    B = mask_pyramid[0].shape[0]
    for (H_l, W_l), m in zip(spatial_shapes.tolist(), mask_pyramid):
        if m.ndim == 2:
            m = m.view(B, H_l, W_l)
        if m.dtype != torch.bool:
            m = (m > 0.5)            # 阈值转为 bool，保持 True=padding
        valid = ~m                  # 现在 ~ 可用了
        H_valid = valid.any(dim=2).sum(dim=1).clamp(min=1)
        W_valid = valid.any(dim=1).sum(dim=1).clamp(min=1)
        ratios.append(torch.stack([W_valid.float()/W_l, H_valid.float()/H_l], dim=-1))

    return torch.stack(ratios, dim=1).to(device)

# ----------------------------------------
# 注意力机制（掩码支持）
# ----------------------------------------

class FFN_Layer(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------------
# 编码器块 & 解码器块
# ----------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.attn = MSDeformAttn(d_model=embed_dim, n_heads=n_heads, n_levels=n_levels, n_points=n_points)
        self.ffn = FFN_Layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, mask=None):
        """
        src: [B, total_tokens, C]   flatten 后的特征拼接
        pos: [B, total_tokens, C]   对应的多尺度位置编码
        reference_points: [B, total_tokens, n_levels, 2]
        spatial_shapes: [n_levels, 2]
        level_start_index: [n_levels]
        mask: [B, total_tokens]  padding mask
        """
        query = src + pos

        attn_out = self.attn(
            query=query,
            reference_points=reference_points,
            input_flatten=src,  # encoder 是自注意力，value = src
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=mask
        )

        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)


        self.cross_attn = MSDeformAttn(
            d_model=embed_dim, n_heads=n_heads, n_levels=n_levels, n_points=n_points
        )

        # FFN 层
        self.ffn = FFN_Layer(embed_dim)

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(
        self,
        queries,                   # [B, num_queries, C]
        memory,                    # [B, ΣHW, C] (encoder 输出)
        query_pos,                 # [B, num_queries, C]
        reference_points,          # [B, num_queries, L, 2]
        spatial_shapes,            # [L, 2]
        level_start_index,         # [L]
        memory_mask=None           # [B, ΣHW]
    ):

        q = k = queries + query_pos
        self_attn_out, _ = self.self_attn(q, k, value=queries)
        queries = queries + self.dropout1(self_attn_out)
        queries = self.norm1(queries)


        # query + query_pos = decoder 的 query 表示
        #不需要memory_pos
        cross_attn_out = self.cross_attn(
            query=queries + query_pos,               # queries
            reference_points=reference_points,       # [B, num_queries, L, 2]
            input_flatten=memory,                    # encoder memory
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=memory_mask
        )
        queries = queries + self.dropout2(cross_attn_out)
        queries = self.norm2(queries)


        queries = queries + self.dropout3(self.ffn(queries))
        queries = self.norm3(queries)

        return queries


# ----------------------------------------
# 编码器 & 解码器
# ----------------------------------------

class Encoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_levels=n_levels,
                n_points=n_points
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                src,                # [B, ΣHW, C] flatten 后的特征
                pos,                # [B, ΣHW, C] 对应位置编码
                reference_points,   # [B, ΣHW, L, 2] 每个 token 的参考点
                spatial_shapes,     # [L, 2] 每层特征图的 H, W
                level_start_index,  # [L] 每层 flatten 起始索引
                mask=None           # [B, ΣHW] padding mask
                ):
        output = src
        for layer in self.layers:
            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mask=mask
            )
        return self.norm(output)


class Decoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_levels=n_levels,
                n_points=n_points
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(embed_dim)  # 官方实现里 decoder 最后会加 norm

    def forward(self,
                queries,              # [B, num_queries, C]
                memory,               # [B, ΣHW, C] encoder 输出
                query_pos,            # [B, num_queries, C]
                memory_pos,           # [B, ΣHW, C] 位置编码
                reference_points,     # [B, num_queries, L, 2] decoder 每层 query 的参考点
                spatial_shapes,       # [L, 2]
                level_start_index,    # [L]
                memory_mask=None      # [B, ΣHW]
                ):
        intermediate = []  # 保存每层输出（用于级联 refine）

        output = queries
        for layer in self.layers:
            output = layer(
                queries=output,
                memory=memory,
                query_pos=query_pos,
                memory_pos=memory_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_mask=memory_mask
            )
            intermediate.append(self.norm(output))  # 每层都 norm 一下（官方实现）

        # 输出所有 decoder 层结果，用于后续多层预测（分类 + 回归头）
        return torch.stack(intermediate, dim=0)  # [num_layers, B, num_queries, C]


# ----------------------------------------
# MLP 边界框预测头
# ----------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < num_layers - 1 else output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ----------------------------------------
# 主模型（接收掩码输入）
# ----------------------------------------

class DeformableDETR(nn.Module):
    def __init__(self,
                 num_queries=300,
                 num_classes=80,
                 num_encoder_layer=6,
                 num_decoder_layer=6,
                 embed_dim=256,
                 n_heads=8,
                 n_levels=5,
                 n_points=4):
        super().__init__()

        self.patch_ebd = PatchEmbed(embed_dim=embed_dim)

        self.pos_embed = MultiScalePosEmbed(embed_dim=embed_dim, num_levels=n_levels)

        self.encoder = Encoder(
            num_layers=num_encoder_layer,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layer,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points
        )

        self.query_embed = DeformableDecoderQueries(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_feature_levels=n_levels
        )

        self.class_embed = nn.ModuleList([
            nn.Linear(embed_dim, num_classes + 1) for _ in range(num_decoder_layer)
        ])
        self.bbox_embed = nn.ModuleList([
            MLP(embed_dim, embed_dim, 4, 3) for _ in range(num_decoder_layer)
        ])

        # 分类 bias 初始化（提升 early recall）
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        for cls_layer in self.class_embed:
            nn.init.constant_(cls_layer.bias, bias_value)

        # 回归头最后层初始化为 0（更稳定）
        for bbox_layer in self.bbox_embed:
            nn.init.zeros_(bbox_layer.layers[-1].weight)
            nn.init.zeros_(bbox_layer.layers[-1].bias)

    def forward(self, x, mask=None, return_all_layers=True):
        bs = x.shape[0]

        src_flatten, spatial_shapes, level_start_index, mask_down, mask_pyramid = self.patch_ebd(x, mask)

        pos_embed = self.pos_embed(spatial_shapes, batch_size=bs, device=x.device)

        if mask_pyramid[0] is None:
            valid_ratios = torch.ones(bs, spatial_shapes.size(0), 2, device=x.device)
        else:
            valid_ratios = get_valid_ratios_from_down_masks(mask_pyramid, spatial_shapes, device=x.device)

        reference_points_enc = get_reference_points(spatial_shapes, valid_ratios, device=x.device)

        memory = self.encoder(
            src=src_flatten,
            pos=pos_embed,
            reference_points=reference_points_enc,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            mask=mask_down
        )  # [B, ΣHW, C]

        queries, query_pos, reference_points_dec = self.query_embed(bs, device=x.device)

        hs = self.decoder(
            queries=queries,
            memory=memory,
            query_pos=query_pos,
            memory_pos=pos_embed,
            reference_points=reference_points_dec,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_mask=mask_down
        )  # [num_layers, B, num_queries, C]

        all_class_logits, all_bbox_preds = [], []
        for i in range(hs.shape[0]):
            out = hs[i]  # [B, num_queries, C]
            class_logits = self.class_embed[i](out)
            bbox_pred = torch.sigmoid(self.bbox_embed[i](out))  # [B, num_queries, 4]
            all_class_logits.append(class_logits)
            all_bbox_preds.append(bbox_pred)

        if return_all_layers:
            return all_class_logits, all_bbox_preds  # list[num_layers][B, Q, ...]
        else:
            return all_class_logits[-1], all_bbox_preds[-1]



# ----------------------------------------
# 测试（传入掩码）
# ----------------------------------------

if __name__ == '__main__':
    model = DeformableDETR(num_queries=100, num_classes=80).cuda()
    model.eval()  # 测试阶段用 eval()，避免dropout干扰

    dummy_input = torch.rand(2, 3, 832, 832).cuda()

    dummy_mask = torch.zeros(2, 832, 832, dtype=torch.bool).cuda()
    pad_size = 2
    dummy_mask[:, :pad_size, :] = True
    dummy_mask[:, -pad_size:, :] = True
    dummy_mask[:, :, :pad_size] = True
    dummy_mask[:, :, -pad_size:] = True

    print("输入掩码 dummy_mask 形状:", dummy_mask.shape)

    with torch.no_grad():
        all_cls, all_bbox = model(dummy_input, mask=dummy_mask, return_all_layers=True)

    print("\nDecoder 层数:", len(all_cls))
    for i, (cls_i, bbox_i) in enumerate(zip(all_cls, all_bbox)):
        print(f"第 {i+1} 层分类预测: {cls_i.shape}  (应为 [B, num_queries, num_classes+1])")
        print(f"第 {i+1} 层边框预测: {bbox_i.shape}  (应为 [B, num_queries, 4])")

    with torch.no_grad():
        last_cls, last_bbox = model(dummy_input, mask=dummy_mask, return_all_layers=False)

    print("\n最后一层分类预测形状:", last_cls.shape)  # [B, num_queries, num_classes + 1]
    print("最后一层边框预测形状:", last_bbox.shape)   # [B, num_queries, 4]

    #print("\n分类 logits 范围:", last_cls.min().item(), "→", last_cls.max().item())
    #print("边框预测范围:", last_bbox.min().item(), "→", last_bbox.max().item())
    #print(model)