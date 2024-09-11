# cryptography.py
import string

def caesar_cipher_encrypt(plaintext, key):
    """
    Encrypts the plaintext using Caesar Cipher with a given key (shift).
    """
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[key:] + alphabet[:key]
    table = str.maketrans(alphabet, shifted_alphabet)
    return plaintext.translate(table)

def caesar_cipher_decrypt(ciphertext, key):
    """
    Decrypts the ciphertext using Caesar Cipher with a given key (shift).
    """
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[-key:] + alphabet[:-key]
    table = str.maketrans(alphabet, shifted_alphabet)
    return ciphertext.translate(table)
