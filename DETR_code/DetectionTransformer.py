import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# detr模型部分
# 使用嵌入维度统一为embed_dim = 256, 6层encoder, 6层decoder

# ----------------------------------------
# 基础模块（添加掩码处理）
# ----------------------------------------
# DETR标准mask方式, 最近邻插值...右下角填充padding=0。掩码矩阵中填充掩码为1, 非填充掩码为0
# 掩码机制:在数据集准备的时候处理好, 指定维度(bs, H, W), H、W是需要输入模型的目标高宽, 掩码是与每张图一一对应的
# 数据集准备时, 处理的图像会在右下角区域添加黑色像素(0), 这些区域的掩码为1, 彩色区域掩码为0

class PatchEmbed(nn.Module):
    """提取特征并生成下采样后的掩码"""
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 输出 (bs, 2048, H/32, W/32), 只取用resnet50前48层, 后2层冻结
        self.conv = nn.Conv2d(2048, embed_dim, kernel_size=1) # (bs, 256, h, w)

    def forward(self, x, mask=None):
        features = self.backbone(x)  # (bs, 2048, H/32, W/32)
        x = self.conv(features)      # (bs, embed_dim, H/32, W/32)
        h, w = x.shape[-2:]          # 获取输出的高度和宽度
        
        # 展平特征图用于 transformer 输入
        x = x.flatten(2).transpose(1, 2)  # (bs, h*w, embed_dim) = (bs, num_tokens=h*w, embed_dim=256)

        # 处理掩码：使用最近邻插值进行下采样
        if mask is not None:
            # 确保 mask 是四维张量 (bs, 1, H, W)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # 添加通道维度 (bs, 1, H, W)

            # 使用最近邻插值下采样到特征图尺寸
            mask_down = F.interpolate(
                mask.float(), 
                size=(h, w), 
                mode='nearest'
            )  # (bs, 1, h, w)

            # 转换为布尔掩码，squeeze 去掉通道维度后展平
            mask_down = mask_down.squeeze(1).flatten(1)  # (bs, h*w)
            mask_down = mask_down.bool()  # False 表示填充位置，True 表示有效位置
            #print(f"有效像素个数：{torch.sum(mask_down==0)}, 填充像素个数： {torch.sum(mask_down==1)}")
            #print(mask_down) # (bs, h*w)
            
        else:
            mask_down = None
        # x=(bs, num_tokens=h*w), h=H/32, w=W/32, mask=(bs, H, W)--->mask_down=(bs, h*w)
        return x, h, w, mask_down
    
    
# encoder部分的二维位置编码PositionEncode
class PositionEncode(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # nn.Embedding  相当于 nn.Parameter  其实就是初始化函数
        self.row_embed = nn.Embedding(50, embed_dim//2) # (50, 128)
        self.col_embed = nn.Embedding(50, embed_dim//2) # (50, 128)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, h, w):
        i = torch.arange(w, device=x.device) # [0, 1, 2, 3, ..., w-1]
        j = torch.arange(h, device=x.device) # [0, 1, 2, 3, ..., h-1]
        x_emb = self.col_embed(i)   # 初始化x方向位置编码, 把self.row_embed中索引为[0, 1, 2, 3, ..., w-1]的初始化元素取出--->(w, 128)
        y_emb = self.row_embed(j)   # 初始化y方向位置编码, 把self.col_embed中索引为[0, 1, 2, 3, ..., h-1]的初始化元素取出--->(h, 128)
        # concat x y 方向位置编码
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # x_emb.unsqueeze(0).repeat(h, 1, 1) = (h, w, 128)
        # y_emb.unsqueeze(0).repeat(h, 1, 1) = (h, w, 128), dim = -1按最后一个维度cat拼接
        # pos=(bs, h, w, 256)
        return pos.flatten(1, 2)  # (bs, num_tokens=h*w, embed_dim=256)

# decoder部分初始化全零查询（不可学习参数）、以及查询对应的位置编码（可学习参数）
class QueryPositionEncode(nn.Module):
    """可学习的查询向量"""
    def __init__(self, num_queries=100, embed_dim=256):
        super().__init__()
        self.queries = torch.zeros(1, num_queries, embed_dim)
        self.query_pos = nn.Embedding(num_queries, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        object_queries = self.queries.expand(batch_size, -1, -1).to(device=x.device) # (bs, num_queries=100, embed_dim=256)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch_size, -1, -1) # (bs, num_queries=100, embed_dim=256)
        return object_queries, query_pos

# ----------------------------------------
# 注意力机制（掩码支持）
# ----------------------------------------
# 自注意力阶段的多头注意力机制
class MultiAttention(nn.Module):
    def __init__(self, embed_dim=256, n_head=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim

        # 合并qk的线性变换
        self.qk = nn.Linear(embed_dim, 2 * embed_dim)
        self.v = nn.Linear(embed_dim,  embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, value, mask=None):
        B, Nq, _ = query.shape
        # qkv ---> [bs, num_tokens=h*w, embed_dim=256], qkv ---> [bs, num_queries=100, embed_dim=256]
        qk = self.qk(query).reshape(B, Nq, self.n_head, 2 * self.head_dim).permute(0, 2, 1, 3)
        q, k = qk.chunk(2, dim=-1)
        v = self.v(value).reshape(B, Nq, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        # qkv--->(bs, n_head=8, tokens=h*w, head_dim=256/8=32) * (bs, n_head=8, 32, tokens)  --->(bs, n_head=8, tokens=h*w, tokens)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # 应用掩码
        if mask is not None:
            # mask.unsqueeze(1).unsqueeze(2), mask由[bs, h*w] ---> [bs, 1, 1, h*w], masked_fill把掩码处为1的填充为-inf
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, self.embed_dim) # (bs, num_tokens=h*w, embed_dim=256)
        
        return self.proj(x)

# 多头交叉注意力机制, 为了方便讲解, 把它单独拿出来做介绍, 其实其结构与多头自注意力机制完全一样
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, n_head=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        B, Nq, _ = query.shape  # (bs, num_queries=100)
        B, Nk, _ = key.shape # (bs, num_tokens=h*w)

        q = self.q(query).reshape(B, Nq, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, 8, num_queries=100, 32)
        k = self.k(key).reshape(B, Nk, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, 8, num_tokens=h*w, 32)
        v = self.v(value).reshape(B, Nk, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (bs, 8, num_tokens=h*w, 32)
        # q*kT = (bs, 8, 100, 32) * (bs, 8, 32, h*w) = (bs, 8, 100, h*w)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # 应用掩码
        if mask is not None:
            # mask=(bs, h*w) --->(bs, 1, 1, h*w)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn = attn.softmax(dim=-1)
        # attn=(bs, 8, 100, h*w), attn*v = (bs, 8, 100, h*w) * (bs, 8, h*w, 32)--->(bs, 8, 100, 32)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, self.embed_dim) # (bs, num_queries=100, 256)
        return self.proj(x)
    
# 前馈神经网络层, 进一步提取特征
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
    def __init__(self, embed_dim=256, n_head=8):
        super().__init__()
        self.attn = MultiAttention(embed_dim, n_head)
        self.ffn = FFN_Layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, pos, mask=None):
        qk = x + pos  # q = k = x + pos, v = x, x是经过backbone下采样32倍后特征提取后的值
        v = x
        attn_out = self.attn(qk, v, mask)  # 多头自注意力 传入掩码mask, mask是经过最近邻插值下采样32倍后的值
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        x = x + self.dropout2(self.ffn(x))
        x = self.norm2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=256, n_head=8):
        super().__init__()
        self.self_attn = MultiAttention(embed_dim, n_head)
        self.cross_attn = CrossAttention(embed_dim, n_head)
        self.ffn = FFN_Layer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, queries, memory, query_pos, memory_pos, mask=None):
        # 自注意力（不需要传入掩码）
        qk_self = queries + query_pos  # q = k = queries + query_pos, v = queries, 不同与自然语言处理, decoder自注意力阶段不需要添加掩码机制
        self_attn_out = self.self_attn(qk_self, queries)
        queries = queries + self.dropout1(self_attn_out)
        queries = self.norm1(queries)
        
        # 交叉注意力（需要传入掩码）
        # q = queries + query_pos, k = memory + memory_pos, v = memory, 添加mask, 就是复用最初特征提取后的mask, 掩蔽黑色部分
        cross_attn_out = self.cross_attn(queries + query_pos, memory + memory_pos, memory, mask)
        queries = queries + self.dropout2(cross_attn_out)
        queries = self.norm2(queries)
        
        # FFN前馈神经网络
        queries = queries + self.dropout3(self.ffn(queries))
        queries = self.norm3(queries)
        return queries # (bs, num_queries=100, embed_dim=256)

# ----------------------------------------
# 编码器 & 解码器
# ----------------------------------------

class Encoder(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(num_layers)])

    def forward(self, x, pos, mask):
        for layer in self.layers:
            x = layer(x, pos, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, queries, memory, query_pos, memory_pos, mask):
        intermediate = []
        for layer in self.layers:
            queries = layer(queries, memory, query_pos, memory_pos, mask)
            intermediate.append(queries)
        return torch.stack(intermediate)  # [L, B, Q, D]

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

class DETR_model(nn.Module):
    def __init__(self, num_queries=100, num_classes=80, num_encoder_layer=6, num_decoder_layer=6):
        super().__init__()
        self.patch_ebd = PatchEmbed()

        self.encoder_pos = PositionEncode()
        self.encoder = Encoder(num_encoder_layer)

        self.query_embed = QueryPositionEncode(num_queries=num_queries)
        self.decoder = Decoder(num_decoder_layer)
        

        # 分类头和边界框头, 是独立的头, 有几层decoder, 就有几个独立头, 用于每一层独立进行匈牙利匹配, 每一层独立进行损失计算, 达到辅助损失的目的
        self.class_embed = nn.ModuleList([
            nn.Linear(256, num_classes + 1) for _ in range(num_decoder_layer)
        ])
        self.bbox_embed = nn.ModuleList([
            MLP(256, 256, 4, 3) for _ in range(num_decoder_layer)
        ])

        # 初始化分类偏置项
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        for cls_layer in self.class_embed:
            nn.init.constant_(cls_layer.bias, bias_value)
        
        # 添加回归头的零初始化（稳定训练）
        for bbox_layer in self.bbox_embed:
            nn.init.zeros_(bbox_layer.layers[-1].weight)  # 最后一层权重初始化为0
            nn.init.zeros_(bbox_layer.layers[-1].bias)    # 偏置初始化为0

    def forward(self, x, mask=None, return_all_layers=True):
        src, h, w, mask_down = self.patch_ebd(x, mask)  # 接收处理后的图像、初始掩码, 进行backbone处理下采样32倍  
        encoder_pos = self.encoder_pos(src, h, w)
        # encoder_pos, mask_down是复用的量
        memory = self.encoder(src, encoder_pos, mask_down)  # 编码器多头自注意力处使用掩码mask_down
        
        queries, queries_pos = self.query_embed(src)
        # memory, queries_pos, encoder_pos, mask_down是复用的量
        decoder_outputs = self.decoder(queries, memory, queries_pos, encoder_pos, mask_down)  # 解码器交叉注意力处使用掩码mask_down
        
        all_class_logits = []
        all_bbox_preds = []
        for i in range(decoder_outputs.size(0)):
            out = decoder_outputs[i]
            class_logits = self.class_embed[i](out)
            bbox_pred = torch.sigmoid(self.bbox_embed[i](out))
            all_class_logits.append(class_logits)
            all_bbox_preds.append(bbox_pred)

        if return_all_layers:
            return all_class_logits, all_bbox_preds
        else:
            return all_class_logits[-1], all_bbox_preds[-1]

# ----------------------------------------
# 测试（传入掩码）
# ----------------------------------------

if __name__ == '__main__':
    model = DETR_model(num_queries=100, num_classes=80).cuda()
    dummy_input = torch.rand(2, 3, 800, 800).cuda()
    
    # 图像掩码机制： 填充区域掩码为1， 非填充区域掩码为0
    # 创建初始掩码, 全0张量（假设有效区域为内部）
    dummy_mask = torch.zeros(2, 800, 800, dtype=torch.bool).cuda()
    # 设置外层2像素为True=1（填充区域）
    pad_size = 2
    dummy_mask[:, :pad_size, :] = 1        # 上边缘
    dummy_mask[:, -pad_size:, :] = 1       # 下边缘
    dummy_mask[:, :, :pad_size] = 1       # 左边缘
    dummy_mask[:, :, -pad_size:] = 1       # 右边缘
    print(dummy_mask)
    
    # 训练模式（返回所有层）
    all_cls, all_bbox = model(dummy_input, mask=dummy_mask)
    print(f"Number of decoder layers: {len(all_cls)}")
    print(f"Shape of first layer class predictions: {all_cls[0].shape}")
    #print(f" first layer class predictions: {all_cls[0]}")
    print(f"Shape of first layer bbox predictions: {all_bbox[0].shape}")

    # 推理模式（返回最后一层）
    last_cls, last_bbox = model(dummy_input, mask=dummy_mask, return_all_layers=False)
    print(f"\nShape of final class predictions: {last_cls.shape}")
    print(f"Shape of final bbox predictions: {last_bbox.shape}")