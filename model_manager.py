import torch
import os
import json
import logging

from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, MODEL_VERSION_FILE, MAX_MODELS_KEPT, DEVICE
from model import ImprovedOrderbookTransformer, INPUT_DIM, MODEL_CONFIG


class ModelManager:
    @staticmethod
    def load_latest_model():
        try:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            if not os.path.exists(MODEL_VERSION_FILE):
                with open(MODEL_VERSION_FILE, 'w') as f:
                    json.dump({'version': 0}, f)
                logging.info(f"Created new model version file: {MODEL_VERSION_FILE}")
                return None, None

            with open(MODEL_VERSION_FILE, 'r') as f:
                version = json.load(f)['version']

            if version == 0:
                logging.info("No trained model found. Starting with a new model.")
                return None, None

            model_path = os.path.join(MODEL_DIR, f'model_v{version}.pth')

            if not os.path.exists(model_path):
                logging.warning(f"Model file not found: {model_path}")
                return None, None

            checkpoint = torch.load(model_path, map_location=DEVICE)
            model = ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)

            # 尝试加载模型参数，忽略不匹配的键
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

            scaler = StandardScaler()
            if checkpoint['scaler_state'] is not None:
                scaler.__setstate__(checkpoint['scaler_state'])
            else:
                logging.warning("Scaler state not found in checkpoint, using uninitialized scaler.")

            logging.info(f"Loaded model version {version}")
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading latest model: {str(e)}")
            return None, None

    @staticmethod
    def load_base_model():
        base_model_path = os.path.join(MODEL_DIR, 'base_model.pth')
        if os.path.exists(base_model_path):
            logging.info(f"Loading base model from {base_model_path}")
            checkpoint = torch.load(base_model_path, map_location=DEVICE)
            model = ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Base model loaded successfully")
            return model
        else:
            logging.warning(f"Base model not found at {base_model_path}. Initializing new model.")
            model = ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)
            logging.info("New model initialized")
            return model

    @staticmethod
    def save_model(model, scaler, is_best=False, is_new=False, is_base=False):
        try:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            if is_new:
                version = 1
            else:
                with open(MODEL_VERSION_FILE, 'r') as f:
                    version_info = json.load(f)
                    version = version_info['version'] + 1

            model_filename = f'model_v{version}.pth'
            model_path = os.path.join(MODEL_DIR, model_filename)

            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_state': scaler.__getstate__() if scaler is not None else None,
                'version': version
            }, model_path)

            with open(MODEL_VERSION_FILE, 'w') as f:
                json.dump({'version': version}, f)

            logging.info(f"Saved model version {version}")

            if is_best:
                best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'version': version
                }, best_model_path)
                logging.info(f"Saved new best model: {best_model_path}")

            if is_base:
                base_model_path = os.path.join(MODEL_DIR, 'base_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'version': version
                }, base_model_path)
                logging.info(f"Saved base model: {base_model_path}")

        except Exception as e:
            logging.error(f"保存模型时发生错误: {str(e)}")

    @staticmethod
    def cleanup_old_models():
        try:
            models = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_v') and f.endswith('.pth')]
            models.sort(key=lambda x: int(x.split('v')[1].split('.')[0]), reverse=True)
            for model in models[MAX_MODELS_KEPT:]:
                os.remove(os.path.join(MODEL_DIR, model))
                logging.info(f"Deleted old model: {model}")
        except Exception as e:
            logging.error(f"Error cleaning up old models: {str(e)}")

    @staticmethod
    def get_model_version():
        try:
            with open(MODEL_VERSION_FILE, 'r') as f:
                return json.load(f)['version']
        except FileNotFoundError:
            return 0
        except Exception as e:
            logging.error(f"Error getting model version: {str(e)}")
            return 0