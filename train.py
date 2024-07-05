import torch
import torch.nn as nn
import numpy as np
import logging
from model_manager import ModelManager
from data_collector import DataCollector
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, N_SAMPLES, SEQUENCE_LENGTH, ONLINE_LEARNING_RATE, \
    ONLINE_LEARNING_WINDOW
from database import db


def retrain_model():
    logging.info("开始重新训练模型...")
    try:
        model = ModelManager.load_base_model()
        data_collector = DataCollector()

        # 收集训练数据
        train_data = data_collector.collect_train_data(N_SAMPLES, SEQUENCE_LENGTH)

        # 准备训练数据
        train_data = np.array(train_data)
        scaler = data_collector.get_scaler()
        train_data_flat = train_data.reshape(-1, train_data.shape[-1])
        train_data_scaled = scaler.fit_transform(train_data_flat).reshape(train_data.shape)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 训练模型
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for i in range(0, N_SAMPLES - SEQUENCE_LENGTH, BATCH_SIZE):
                batch = train_data_scaled[i:i + BATCH_SIZE]
                inputs = torch.FloatTensor(batch[:, :-1]).to(DEVICE)
                targets = torch.FloatTensor(batch[:, -1, 0] - batch[:, -2, 0]).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (N_SAMPLES // BATCH_SIZE)
            logging.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

        # 保存新模型
        ModelManager.save_model(model, scaler)
        logging.info("模型重新训练完成并保存")

        # 验证新模型
        validation_data = data_collector.collect_validation_data()
        old_loss = validate_model(ModelManager.load_latest_model()[0], validation_data)
        new_loss = validate_model(model, validation_data)

        if new_loss < old_loss:
            ModelManager.save_model(model, scaler, is_best=True)
            logging.info("新模型性能更好，已保存为最佳模型")
        else:
            logging.info("新模型性能未改善，保留旧模型")

    except Exception as e:
        logging.error(f"重新训练模型时发生错误: {str(e)}", exc_info=True)


def validate_model(model, validation_data):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        inputs, targets = validation_data
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
    return loss.item()


def update_model_with_predictions():
    logging.info("开始在线学习...")
    try:
        model, scaler = ModelManager.load_latest_model()
        if model is None or scaler is None:
            logging.error("无法加载最新模型，在线学习失败")
            return

        # 获取最近的预测结果
        recent_predictions = db.get_recent_predictions(ONLINE_LEARNING_WINDOW)
        if len(recent_predictions) < 2:
            logging.info("没有足够的预测数据进行在线学习")
            return

        # 准备在线学习数据
        inputs = []
        targets = []
        for i in range(len(recent_predictions) - 1):
            current_pred = recent_predictions[i]
            next_pred = recent_predictions[i + 1]

            input_data = [
                current_pred['current_price'],
                current_pred['predicted_change'],
                current_pred['sma']
            ]
            inputs.append(input_data)

            actual_change = next_pred['current_price'] - current_pred['current_price']
            targets.append(actual_change)

        inputs = np.array(inputs)
        targets = np.array(targets)

        # 标准化数据
        inputs_scaled = scaler.transform(inputs)

        # 在线学习
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=ONLINE_LEARNING_RATE)
        criterion = nn.MSELoss()

        inputs_tensor = torch.FloatTensor(inputs_scaled).to(DEVICE)
        targets_tensor = torch.FloatTensor(targets).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs_tensor)
        loss = criterion(outputs.squeeze(), targets_tensor)
        loss.backward()
        optimizer.step()

        logging.info(f"在线学习完成，损失: {loss.item()}")

        # 保存更新后的模型
        ModelManager.save_model(model, scaler)
        logging.info("更新后的模型已保存")

    except Exception as e:
        logging.error(f"在线学习过程中发生错误: {str(e)}", exc_info=True)


if __name__ == "__main__":
    retrain_model()