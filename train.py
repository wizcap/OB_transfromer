import torch
import torch.nn as nn
import numpy as np
import logging
from model_manager import ModelManager
from data_collector import DataCollector
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, N_SAMPLES, SEQUENCE_LENGTH
import os


def retrain_model():
    logging.info("开始重新训练模型...")
    try:
        model = ModelManager.load_base_model()
        data_collector = DataCollector()

        logging.info(f"开始收集训练数据，共 {N_SAMPLES} 个样本...")
        train_data = data_collector.collect_train_data(N_SAMPLES, SEQUENCE_LENGTH)
        logging.info("训练数据收集完成")

        if not train_data:
            logging.error("没有收集到训练数据")
            return

        train_data = np.array(train_data)
        scaler = data_collector.get_scaler()
        train_data_flat = train_data.reshape(-1, train_data.shape[-1])
        train_data_scaled = scaler.fit_transform(train_data_flat).reshape(train_data.shape)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        logging.info(f"开始训练，总共 {NUM_EPOCHS} 个 epochs")
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            batch_count = 0
            for i in range(0, len(train_data_scaled), BATCH_SIZE):
                batch = train_data_scaled[i:i + BATCH_SIZE]
                if len(batch) < 2:  # 确保至少有两个样本来计算差异
                    continue
                inputs = torch.FloatTensor(batch[:, :-1]).to(DEVICE)
                targets = torch.FloatTensor(batch[:, -1, 0] - batch[:, -2, 0]).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                logging.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')
            else:
                logging.warning(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], No valid batches')

        ModelManager.save_model(model, scaler)
        logging.info("模型重新训练完成并保存")

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