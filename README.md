# 加密货币价格预测系统 / Cryptocurrency Price Prediction System

## 中文说明

### 项目概述
这是一个基于机器学习的加密货币价格预测系统。该系统使用订单簿数据和历史价格信息,结合改进的Transformer模型和XGBoost算法,来预测短期的价格变动。

### 主要功能
1. 数据收集: 从Binance交易所实时获取订单簿和价格数据。
2. 价格预测: 使用改进的Transformer模型预测短期价格变动。
3. 趋势分析: 结合XGBoost算法进行趋势判断。
4. 模型训练: 定期重新训练模型以适应市场变化。
5. 预测评估: 定期评估预测准确性。

### 技术栈
- Python 3.8+
- PyTorch: 用于深度学习模型
- XGBoost: 用于辅助预测和趋势分析
- SQLite: 用于数据存储
- ccxt: 用于与交易所API交互

### 安装和使用
1. 克隆仓库:
   ```
   git clone [repository_url]
   ```
2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```
3. 配置 `config.py` 文件,设置相关参数。
4. 运行主程序:
   ```
   python main.py
   ```

### 文件结构
- `main.py`: 主程序入口
- `data_collector.py`: 数据收集模块
- `model.py`: 神经网络模型定义
- `predict.py`: 预测逻辑实现
- `train.py`: 模型训练逻辑
- `evaluate.py`: 预测评估模块
- `database.py`: 数据库操作
- `config.py`: 配置文件

### 注意事项
- 本系统仅用于研究和学习目的,不应用于实际交易决策。
- 请确保遵守相关法律法规和交易所的使用政策。

## English Description

### Project Overview
This is a machine learning-based cryptocurrency price prediction system. The system uses order book data and historical price information, combined with an improved Transformer model and XGBoost algorithm, to predict short-term price movements.

### Key Features
1. Data Collection: Real-time order book and price data retrieval from Binance exchange.
2. Price Prediction: Short-term price movement prediction using an improved Transformer model.
3. Trend Analysis: Trend determination combining XGBoost algorithm.
4. Model Training: Periodic model retraining to adapt to market changes.
5. Prediction Evaluation: Regular evaluation of prediction accuracy.

### Tech Stack
- Python 3.8+
- PyTorch: For deep learning models
- XGBoost: For auxiliary prediction and trend analysis
- SQLite: For data storage
- ccxt: For interacting with exchange APIs

### Installation and Usage
1. Clone the repository:
   ```
   git clone [repository_url]
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the `config.py` file with relevant parameters.
4. Run the main program:
   ```
   python main.py
   ```

### File Structure
- `main.py`: Main program entry
- `data_collector.py`: Data collection module
- `model.py`: Neural network model definition
- `predict.py`: Prediction logic implementation
- `train.py`: Model training logic
- `evaluate.py`: Prediction evaluation module
- `database.py`: Database operations
- `config.py`: Configuration file

### Important Notes
- This system is for research and learning purposes only and should not be used for actual trading decisions.
- Ensure compliance with relevant laws, regulations, and exchange usage policies.
