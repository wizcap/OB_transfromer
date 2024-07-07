import logging
import threading
from database import init_db, close_db
from sklearn.preprocessing import StandardScaler

from scheduler import start_scheduler
from train import retrain_model
import os
from config import MODEL_DIR, DATA_DIR, DEVICE
from model import ImprovedOrderbookTransformer, INPUT_DIM, MODEL_CONFIG
from model_manager import ModelManager

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log", encoding="utf-8"),
                        logging.StreamHandler()
                    ])


def user_input_handler():
    while True:
        command = input("输入 'retrain' 来重新训练模型: ")
        if command.lower() == 'retrain':
            print("开始手动重新训练...")
            retrain_model()
            print("手动重新训练完成。")


def initialize_base_model():
    if not os.path.exists(os.path.join(MODEL_DIR, 'base_model.pth')):
        logging.info("初始化并保存基础模型...")
        model = ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)
        ModelManager.save_model(model, StandardScaler(), is_best=False, is_base=True)
        logging.info("基础模型已初始化并保存")


def main():
    init_db()
    # 确保必要的目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 初始化基础模型
    initialize_base_model()
    logging.info("程序开始执行")
    try:
        # 启动用户输入处理线程
        threading.Thread(target=user_input_handler, daemon=True).start()
        logging.info("用户输入处理线程启动完成")

        # 启动调度器
        logging.info("开始定时任务...")
        start_scheduler()

    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {str(e)}", exc_info=True)
    finally:
        close_db()


if __name__ == "__main__":
    main()
