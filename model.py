import torch
import torch.nn as nn
import numpy as np
from config import INPUT_DIM, MODEL_CONFIG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ImprovedOrderbookTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(ImprovedOrderbookTransformer, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Tanh()  # 使用tanh函数限制输出范围在[-1, 1]之间
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.feature_extractor(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.dropout(output)
        output = self.decoder(output[:, -1, :])
        return output


# 这里我们从配置文件中导入模型配置
INPUT_DIM = INPUT_DIM
MODEL_CONFIG = MODEL_CONFIG

# 确保类被正确导出
__all__ = ['ImprovedOrderbookTransformer', 'INPUT_DIM', 'MODEL_CONFIG']
