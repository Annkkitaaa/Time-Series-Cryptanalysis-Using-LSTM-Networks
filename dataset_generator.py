# dataset_generator.py
import random
import string
import pandas as pd
from cryptography import caesar_cipher_encrypt

def generate_random_text(length=10):
    """
    Generates a random string of lowercase letters.
    """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_ciphertext_series(n_samples=1000, key=3):
    """
    Generates a series of random plaintexts and their corresponding ciphertexts.
    """
    data = []
    for _ in range(n_samples):
        plaintext = generate_random_text(10)
        ciphertext = caesar_cipher_encrypt(plaintext, key)
        data.append((plaintext, ciphertext))

    df = pd.DataFrame(data, columns=["plaintext", "ciphertext"])
    df.to_csv("data/ciphertexts.csv", index=False)
    print(f"Generated {n_samples} samples.")
    return df
