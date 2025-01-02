import json
import logging
from pathlib import Path

class ConfigManager:
    """Manages application configuration"""
    DEFAULT_CONFIG = {
        'min_face_size': (30, 30),
        'detection_confidence': 0.5,
        'feature_extraction': {
            'image_size': (224, 224),
            'batch_size': 32
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        },
        'camera': {
            'capture_interval': 0.1,
            'frame_size': (640, 480)
        }
    }

    def __init__(self, config_path: Path = Path('config/config.json')):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            return self.DEFAULT_CONFIG
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.DEFAULT_CONFIG

    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
