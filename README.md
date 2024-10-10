
# 🔮 Next Word Prediction using LSTM

## 📚 Introduction
This project implements a **Next Word Prediction model** using **Long Short-Term Memory (LSTM)** neural networks. The model predicts the most likely next word in a sequence of words based on previous inputs. This project is designed to assist writers and content creators by providing real-time word suggestions, improving productivity and writing flow.

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/y/Next-Word-Prediction-using-LSTM.git
   cd next-word-prediction
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pretrained Model and Tokenizer**:
   Ensure you have the following files in the project directory:
   - `next_word_prediction.h5` (the trained LSTM model)
   - `tokenizer.pickle` (the fitted tokenizer)

---

## 🚀 Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   
2. **Input Text**:
   Begin typing a sentence in the input box, and the app will suggest the next word along with prediction confidence.

---

## 🌟 Features

- **Real-Time Next Word Prediction**: LSTM model predicts the next word based on the given input text.
- **Top N Predictions**: Displays the top 5 predictions with their associated confidence scores.
- **Responsive UI**: An interactive, user-friendly interface built using Streamlit, designed for smooth user experience.
- **Customizable**: Ability to fine-tune the model and tokenizer for various text corpora.

---

## 📊 Data

- **Source**: The model is trained on text data from Shakespeare's *Hamlet* (sourced from the Gutenberg corpus).
- **Preprocessing**: The text is tokenized, sequences are padded, and data is split into predictors (X) and labels (Y) for training.

---

## 🔬 Methodology

1. **Tokenization**: Text is tokenized into sequences using the Keras `Tokenizer`.
2. **Padding**: Input sequences are padded to ensure consistent length.
3. **Model Architecture**: 
   - Embedding layer with 100-dimensional vectors.
   - Two LSTM layers (150 units and 100 units) with Dropout to prevent overfitting.
   - A Dense layer with softmax activation to predict the next word.
4. **Training**: The model is trained using categorical cross-entropy loss and the Adam optimizer.

---

## 📈 Results

The model achieves accurate next-word predictions, offering a user-friendly solution to assist with writing and text completion tasks. The top N predictions are presented with confidence levels to give users insight into the model's decisions.

---

## 🎯 Conclusion

The **Next Word Prediction using LSTM** project showcases the application of deep learning in language models. It has practical applications in enhancing writing tools, building AI-driven chatbots, and improving user experience in text-based applications.

---

## 🚀 Future Work

- **Dataset Expansion**: Incorporate more diverse text data to improve generalization.
- **Model Fine-Tuning**: Explore Transformer-based architectures for more sophisticated predictions.
- **UI Enhancements**: Add features like dark mode, real-time typing feedback, and interactive charts.

---

## 🤝 Contributing

We welcome contributions! Here's how you can get involved:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Submit a pull request and we'll review it.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---