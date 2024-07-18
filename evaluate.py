import logging
from database import db
from config import TREND_THRESHOLD


def evaluate_predictions():
    logging.info("开始评估预测...")
    try:
        predictions = db.get_recent_predictions(limit=1000)  # 获取最近1000条预测

        correct_predictions = 0
        total_predictions = 0

        for i in range(len(predictions) - 1):
            current_prediction = predictions[i]
            next_prediction = predictions[i + 1]

            predicted_trend = current_prediction['prediction_trend']
            actual_change = next_prediction['market_price'] - current_prediction['market_price']

            # 评估预测是否正确
            if (predicted_trend in ["强烈上涨", "可能上涨"] and actual_change > TREND_THRESHOLD) or \
                    (predicted_trend in ["强烈下跌", "可能下跌"] and actual_change < -TREND_THRESHOLD) or \
                    (predicted_trend == "横盘整理" and abs(actual_change) <= TREND_THRESHOLD):
                correct_predictions += 1
            total_predictions += 1

        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logging.info(f"模型准确率: {accuracy:.2%}")

        return accuracy

    except Exception as e:
        logging.error(f"评估预测时发生错误: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    evaluate_predictions()
