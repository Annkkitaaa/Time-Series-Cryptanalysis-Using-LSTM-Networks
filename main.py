# main.py
from dataset_generator import generate_ciphertext_series, generate_random_text
from lstm_model import load_data, preprocess_data, train_model
from cryptography import caesar_cipher_encrypt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def test_model(model, tokenizer, key=3):
    """
    Tests the trained LSTM model with a new ciphertext.
    """
    # Generate a new random plaintext
    plaintext = generate_random_text()
    # Encrypt the plaintext using Caesar Cipher with a known key
    ciphertext = caesar_cipher_encrypt(plaintext, key)
    print(f"Original Plaintext: {plaintext}")
    print(f"Encrypted Ciphertext: {ciphertext}")

    # Preprocess the ciphertext (tokenize and pad)
    ciphertext_seq = tokenizer.texts_to_sequences([ciphertext])
    X = pad_sequences(ciphertext_seq, maxlen=10, padding='post')

    # Predict the plaintext using the trained model
    predicted_seq = model.predict(X)
    predicted_text = ''.join([tokenizer.index_word[i] for i in predicted_seq.argmax(axis=-1)[0]])

    print(f"Predicted Plaintext: {predicted_text}")

def main():
    # Step 1: Generate data
    generate_ciphertext_series(n_samples=10000, key=3)

    # Step 2: Load and preprocess data
    plaintexts, ciphertexts = load_data()
    X, y, tokenizer = preprocess_data(plaintexts, ciphertexts)

    # Step 3: Train the LSTM model
    model = train_model(X, y, tokenizer, epochs=10)
    print("Model trained and saved successfully.")

    # Step 4: Load the model
    model = load_model("lstm_cryptanalysis_model.h5")

    # Step 5: Test the model with a new ciphertext
    test_model(model, tokenizer, key=3)

if __name__ == "__main__":
    main()
