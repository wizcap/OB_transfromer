import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置
INPUT_DIM = 47  # 40 (原始特征) + 7 (额外特征)
MODEL_CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'output_dim': 1,
    'dropout': 0.1
}

# 数据收集配置
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'
ORDERBOOK_LIMIT = 10
SEQUENCE_LENGTH = 100

# 训练配置
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
N_SAMPLES = 1000

# 预测配置
PREDICTION_INTERVAL = 5  # 每5分钟进行一次预测

# 评估配置
EVALUATION_INTERVAL = 60  # 每60分钟进行一次评估

# 模型管理配置
MAX_MODELS_KEPT = 5  # 保留的最大模型数量

# 文件路径
MODEL_DIR = 'models/'
DATA_DIR = 'data/'
PREDICTION_RESULTS_FILE = 'data/prediction_results.json'
MODEL_VERSION_FILE = 'model_version.json'

# 在线学习配置
ONLINE_LEARNING_INTERVAL = 120  # 每120分钟进行一次在线学习
ONLINE_LEARNING_WINDOW = 100  # 使用最近100次预测进行在线学习
ONLINE_LEARNING_RATE = 0.0001

# 阈值设置
TREND_THRESHOLD = 0.001  # 0.1% 的阈值用于判断横盘
