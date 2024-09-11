
# **Time-Series Cryptanalysis Using LSTM Networks**

This project demonstrates how to use LSTM (Long Short-Term Memory) networks for cryptanalysis. Specifically, it focuses on breaking the Caesar Cipher encryption, a classical substitution cipher, by predicting the plaintext from ciphertext using a deep learning model.

## **Project Structure**

```
cryptanalysis-lstm/
│
├── data/                             # Directory to hold the generated dataset
│   └── .gitignore                    # Ignores the generated CSV files
├── cryptography.py                   # Contains the Caesar Cipher encryption logic
├── dataset_generator.py              # Generates random plaintext and ciphertext pairs
├── lstm_model.py                     # Builds and trains the LSTM model
├── main.py                           # Main script for running the entire process
├── requirements.txt                  # Contains the list of dependencies
└── README.md                         # Project documentation and setup instructions
```

## **Features**

- **Caesar Cipher Encryption**: Implements Caesar Cipher to generate encrypted text from random plaintext.
- **LSTM-based Decryption**: Uses an LSTM model to predict the original plaintext from ciphertext.
- **Data Generation**: Generates random plaintext-ciphertext pairs for training.
- **Model Training and Evaluation**: Provides a pipeline for training the LSTM model and evaluating it on new ciphertexts.

## **Getting Started**

### **1. Clone the Repository**
First, clone this repository to your local machine:
```bash
git clone https://github.com/Annkkitaaa/Time-Series-Cryptanalysis-Using-LSTM-Networks/cryptanalysis-lstm.git
cd cryptanalysis-lstm
```

### **2. Set Up a Python Virtual Environment**
It is recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
```

### **3. Install Dependencies**
Install the required Python libraries listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **4. Run the Project**
Run the main script to generate data, train the model, and test it:
```bash
python main.py
```

The script will:
- Generate random plaintext-ciphertext pairs.
- Train the LSTM model to predict plaintexts from the corresponding ciphertexts.
- Test the trained model on new ciphertext and print the results.

## **How It Works**

### **1. Data Generation**
The `dataset_generator.py` script generates random strings (plaintexts) and encrypts them using a Caesar Cipher with a given key. These plaintext-ciphertext pairs are saved in a CSV file for training the LSTM model.

### **2. LSTM Model**
The LSTM model is implemented in `lstm_model.py`. It is designed to learn the relationship between the encrypted (ciphertext) and original (plaintext) data. The model is trained on the generated data and is then tested on new ciphertext samples to predict the original plaintext.

### **3. Testing the Model**
After the model is trained, it is saved in an HDF5 file. The `main.py` script tests the model by generating a new ciphertext and predicting the corresponding plaintext.

## **Example Output**

After running the project, you should see output like the following:

```
Model trained and saved successfully.
Original Plaintext: helloworld
Encrypted Ciphertext: khoorzruog
Predicted Plaintext: helloworld
```

This indicates that the model successfully decrypted the ciphertext and predicted the correct plaintext.

## **Project Files**

- **`cryptography.py`**: Contains the Caesar Cipher encryption algorithm.
- **`dataset_generator.py`**: Responsible for generating random plaintexts, encrypting them using the Caesar Cipher, and saving the data to a CSV file.
- **`lstm_model.py`**: Builds and trains an LSTM model using Keras and TensorFlow.
- **`main.py`**: Orchestrates data generation, model training, and testing.

## **Customizations**

### **1. Change Cipher Key**
You can modify the encryption key by adjusting the `key` parameter in the `generate_ciphertext_series()` function in `main.py` and `test_model()` function.

### **2. Test on Custom Ciphertexts**
You can manually generate custom ciphertexts and test them using the trained model by modifying the `test_model()` function in `main.py`.

### **3. Modify Model Architecture**
The LSTM model architecture can be tuned by editing the `train_model()` function in `lstm_model.py`. You can add layers, change the number of units, or modify the optimizer for experimentation.

## **Dependencies**

The project depends on the following libraries:
- **TensorFlow**: 2.9.0+
- **Keras**: (Bundled with TensorFlow)
- **NumPy**: For numerical operations.
- **Pandas**: For handling datasets.
- **scikit-learn**: For pre-processing.

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

## **Future Enhancements**

Here are some possible enhancements for the project:
- Support for additional ciphers (e.g., Vigenère, XOR).
- Implement adversarial attacks on the model to evaluate robustness.
- Expand the model to handle larger datasets and more complex encryption schemes.
- Add attention mechanisms to the LSTM model for improved performance.

 

