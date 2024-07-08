import ccxt
import numpy as np
import torch
import logging
import time
from datetime import datetime
from model_manager import ModelManager
from data_collector import DataCollector
from config import DEVICE, SYMBOL, TREND_THRESHOLD
from database import db, close_db

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'  # 使用期货市场
    }
})


def run_prediction():
    logging.info("开始预测...")
    try:
        model, scaler = ModelManager.load_latest_model()
        if model is None or scaler is None:
            logging.error("无法加载模型，预测失败")
            return None

        data_collector = DataCollector()
        data, _ = data_collector.prepare_data()

        if not hasattr(scaler, 'n_features_in_'):
            logging.warning("Scaler not fitted, fitting with current data")
            scaler.fit(data)
            ModelManager.save_model(model, scaler)  # 保存拟合后的scaler

        data_scaled = scaler.transform(data)

        # 进行预测
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(DEVICE)
            prediction = model(input_tensor).item()

        if not np.isfinite(prediction):
            logging.error(f"预测结果无效: {prediction}")
            return None

        logging.info(f"预测的价格变动: {prediction:.4f}")

        # 获取当前市场数据
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        logging.info(f"当前 {SYMBOL} 价格: {current_price}")

        # 计算预测价格
        predicted_price = current_price + prediction
        logging.info(f"5分钟后预测的 {SYMBOL} 价格: {predicted_price:.2f}")

        # 获取历史数据进行简单的技术分析
        ohlcv = exchange.fetch_ohlcv(SYMBOL, '5m', limit=12)  # 获取过去1小时的5分钟K线数据
        closes = [x[4] for x in ohlcv]
        sma = sum(closes) / len(closes)

        # 判断预测趋势
        if prediction > TREND_THRESHOLD and current_price > sma:
            trend = "强烈上涨"
        elif prediction > TREND_THRESHOLD and current_price <= sma:
            trend = "可能上涨"
        elif prediction < -TREND_THRESHOLD and current_price < sma:
            trend = "强烈下跌"
        elif prediction < -TREND_THRESHOLD and current_price >= sma:
            trend = "可能下跌"
        else:
            trend = "横盘整理"

        logging.info(f"预测趋势：{trend}")

        # 保存预测结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'predicted_change': prediction,
            'predicted_price': predicted_price,
            'market_price': exchange.fetch_ticker(SYMBOL)['last'],
            'prediction_trend': trend,
            'sma': sma
        }

        if all(np.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
            db.save_prediction(result)
            logging.info("预测结果已保存")
        else:
            logging.error("预测结果包含无效值，未保存")

        return result

    except Exception as e:
        logging.error(f"预测过程中发生错误: {str(e)}", exc_info=True)
        return None
    finally:
        close_db()


def continuous_prediction(interval=300):  # 默认每5分钟预测一次
    while True:
        try:
            run_prediction()
            time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("预测被用户中断")
            break
        except Exception as e:
            logging.error(f"连续预测过程中发生错误: {str(e)}", exc_info=True)
            time.sleep(60)  # 如果发生错误，等待1分钟后继续


if __name__ == "__main__":
    continuous_prediction()
