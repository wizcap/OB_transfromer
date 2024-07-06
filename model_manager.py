import torch
import os
import json
import logging
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
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint['scaler']

            logging.info(f"Loaded model version {version}")
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading latest model: {str(e)}")
            return None, None

    @staticmethod
    def load_base_model():
        base_model_path = os.path.join(MODEL_DIR, 'base_model.pth')
        if os.path.exists(base_model_path):
            checkpoint = torch.load(base_model_path, map_location=DEVICE)
            model = ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            logging.warning("Base model not found. Initializing new model.")
            return ImprovedOrderbookTransformer(INPUT_DIM, **MODEL_CONFIG).to(DEVICE)

    @staticmethod
    def save_model(model, scaler, is_best=False):
        try:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            with open(MODEL_VERSION_FILE, 'r') as f:
                version_info = json.load(f)
                current_version = version_info['version']

            new_version = current_version + 1
            model_filename = f'model_v{new_version}.pth'
            model_path = os.path.join(MODEL_DIR, model_filename)

            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'version': new_version
            }, model_path)

            with open(MODEL_VERSION_FILE, 'w') as f:
                json.dump({'version': new_version}, f)

            if is_best:
                best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'version': new_version
                }, best_model_path)
                logging.info(f"Saved new best model: {best_model_path}")

            logging.info(f"Saved model version {new_version}")
            ModelManager.cleanup_old_models()
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")

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