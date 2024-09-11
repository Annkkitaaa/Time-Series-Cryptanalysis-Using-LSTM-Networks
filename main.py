# main.py
from dataset_generator import generate_ciphertext_series
from lstm_model import load_data, preprocess_data, train_model

def main():
    # Step 1: Generate data
    generate_ciphertext_series(n_samples=10000, key=3)

    # Step 2: Load and preprocess data
    plaintexts, ciphertexts = load_data()
    X, y, tokenizer = preprocess_data(plaintexts, ciphertexts)

    # Step 3: Train the LSTM model
    model = train_model(X, y, tokenizer, epochs=10)

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
