# scheduler.py

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR  # 添加这行
from predict import run_prediction
from train import retrain_model, update_model_with_predictions
from evaluate import evaluate_predictions
import logging
from config import PREDICTION_INTERVAL, EVALUATION_INTERVAL, ONLINE_LEARNING_INTERVAL


def start_scheduler():
    scheduler = BlockingScheduler()

    # 添加预测任务
    scheduler.add_job(run_prediction, 'interval', minutes=PREDICTION_INTERVAL,
                      id='prediction_job', name='Prediction Job')
    logging.info(f"预测任务已添加，每 {PREDICTION_INTERVAL} 分钟执行一次")

    # 添加评估任务
    scheduler.add_job(evaluate_predictions, 'interval', minutes=EVALUATION_INTERVAL,
                      id='evaluation_job', name='Evaluation Job')
    logging.info(f"评估任务已添加，每 {EVALUATION_INTERVAL} 分钟执行一次")

    # 添加模型重训练任务
    scheduler.add_job(retrain_model, 'cron', hour=3,
                      id='retrain_job', name='Retrain Job')
    logging.info("模型重训练任务已添加，每天凌晨3点执行")

    # 添加在线学习任务
    scheduler.add_job(update_model_with_predictions, 'interval', minutes=ONLINE_LEARNING_INTERVAL,
                      id='online_learning_job', name='Online Learning Job')
    logging.info(f"在线学习任务已添加，每 {ONLINE_LEARNING_INTERVAL} 分钟执行一次")

    # 错误处理
    def job_listener(event):
        if event.exception:
            logging.error(f'任务 {event.job_id} 执行失败: {str(event.exception)}')

    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    try:
        logging.info("调度器启动")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("调度器被用户中断")
    except Exception as e:
        logging.error(f"调度器发生错误: {str(e)}")
    finally:
        if scheduler.running:
            scheduler.shutdown()
        logging.info("调度器已关闭")


if __name__ == "__main__":
    start_scheduler()