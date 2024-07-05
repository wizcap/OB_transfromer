import logging
import threading
from scheduler import start_scheduler
from train import retrain_model

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


def main():
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


if __name__ == "__main__":
    main()
