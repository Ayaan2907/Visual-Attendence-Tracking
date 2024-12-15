import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from base64 import b64encode, b64decode

def generate_and_save_key(key_path: str = 'encryption_key.key'):
    """Generate and save a new encryption key"""
    try:
        # Generate a random salt
        salt = os.urandom(16)
        
        # Generate a key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Take user input for the secure phrase
        password = input("Enter a secure password: ").encode()
        
        # Create a secure password key
        key = b64encode(kdf.derive(password))
        
        # Create a Fernet key
        fernet_key = Fernet.generate_key()
        
        # Save both salt and keys
        with open(key_path, 'wb') as key_file:
            key_file.write(salt + key + fernet_key)
            
        print(f"Key successfully saved to {key_path}")
        return fernet_key
    except Exception as e:
        print(f"Error generating key: {e}")
        return None

# Generate and save the key
if __name__ == "__main__":
    key = generate_and_save_key()
    if key:
        print(f"Generated key: {key.decode()}")