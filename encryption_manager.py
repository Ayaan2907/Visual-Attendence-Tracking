import os
import cv2
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from pathlib import Path
import logging

class EncryptionManager:
    def __init__(self, key_path: Path = Path('encryption_key.key')):
        """Initialize encryption manager with key handling
        Args:
            key_path (Path): Path to store/retrieve encryption key
        """
        self.key_path = key_path
        self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)
        self.backend = default_backend()
        
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate new one"""
        try:
            if self.key_path.exists():
                with open(self.key_path, 'rb') as key_file:
                    return key_file.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_path, 'wb') as key_file:
                    key_file.write(key)
                return key
        except Exception as e:
            logging.error(f"Error handling encryption key: {e}")
            raise

    def encrypt_image(self, image: np.ndarray) -> bytes:
        """Encrypt image data
        Args:
            image (np.ndarray): Image array to encrypt
        Returns:
            bytes: Encrypted image data
        """
        try:
            # Convert image to bytes
            _, img_buffer = cv2.imencode('.jpg', image)
            img_bytes = img_buffer.tobytes()
            
            # Generate random IV
            iv = os.urandom(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key[:32]),  # Use first 32 bytes as AES key
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Pad data to be multiple of 16 bytes
            pad_length = 16 - (len(img_bytes) % 16)
            padded_data = img_bytes + (bytes([pad_length]) * pad_length)
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            return iv + encrypted_data
            
        except Exception as e:
            logging.error(f"Error encrypting image: {e}")
            raise

    def decrypt_image(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt image data
        Args:
            encrypted_data (bytes): Encrypted image data
        Returns:
            np.ndarray: Decrypted image array
        """
        try:
            # Extract IV and encrypted data
            iv = encrypted_data[:16]
            encrypted_image = encrypted_data[16:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key[:32]),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_data = decryptor.update(encrypted_image) + decryptor.finalize()
            
            # Remove padding
            pad_length = padded_data[-1]
            img_bytes = padded_data[:-pad_length]
            
            # Convert back to image
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)  
        except Exception as e:
            logging.error(f"Error decrypting image: {e}")
            raise

    def secure_save_image(self, image: np.ndarray, file_path: Path) -> None:
        """Save encrypted image to file
        Args:
            image (np.ndarray): Image to save
            file_path (Path): Path to save encrypted image
        """
        try:
            encrypted_data = self.encrypt_image(image)
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            logging.info(f"Securely saved image to {file_path}")
        except Exception as e:
            logging.error(f"Error saving encrypted image: {e}")
            raise

    def secure_load_image(self, file_path: Path) -> np.ndarray:
        """Load and decrypt image from file        
        Args:
            file_path (Path): Path to encrypted image
        Returns:
            np.ndarray: Decrypted image array
        """
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            return self.decrypt_image(encrypted_data)
        except Exception as e:
            logging.error(f"Error loading encrypted image: {e}")
            raise
