# Assignment 8 — Sentiment Analysis using LSTM

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Analyze customer vehicle feedback text and technical vehicle telemetry/specifications using a **Long Short-Term Memory (LSTM)** neural network architecture in PyTorch for multi-class sentiment classification (**Negative**, **Neutral**, **Positive**).

## 2. Dataset & Data Leakage Prevention
- **Dataset:** 450 unique synthetic records (~150 Positive customer feedback, ~150 Neutral technical vehicle specifications/telemetry, ~150 Negative customer feedback).
- **Data Leakage Prevention:** All duplicate sentences were removed prior to splitting (`assert len(set(df['text'])) == len(df)`).
- **Train/Test Splitting:** Stratified 80/20 train-test split (360 train samples, 90 test samples) with zero text overlap between train and test sets (`assert len(set(train_text).intersection(set(test_text))) == 0`).
- **Text Preprocessing & Left-Padding:** Lowercasing, punctuation removal, word tokenization building a 262-token vocabulary (including `<PAD> = 0`), and **Left-Padding** (`MAX_LEN = 15`). Left-padding (`[0, 0, ..., w1, w2]`) ensures the PyTorch LSTM hidden state `hn[-1]` / `lstm_out[:, -1, :]` processes active word vectors instead of zero-padding tokens.

## 3. Model Architecture
- **Embedding Layer:** `nn.Embedding(vocab_size=262, embed_dim=64, padding_idx=0)`
- **LSTM Layer:** `nn.LSTM(embed_dim=64, hidden_dim=64, batch_first=True)`
- **Dropout Layer:** `nn.Dropout(p=0.2)`
- **Linear Output Layer:** `nn.Linear(64, 3)`
- **Optimizer & Loss:** Adam (`lr=0.002`), `nn.CrossEntropyLoss()`.
- **Epochs & Batch Size:** 15 training epochs, Batch Size = 32.

## 4. Actual Executed Results
All values below come directly from the executed notebook (`TTL_Assignment_08.ipynb`):

- **Final Test Accuracy:** **100.00%** (90 / 90 correct test predictions).
- **Precision / Recall / F1-Score:**
  - **Negative (0):** Precision = 1.0000 | Recall = 1.0000 | $F_1$-Score = 1.0000 (Support: 30)
  - **Neutral (1):** Precision = 1.0000 | Recall = 1.0000 | $F_1$-Score = 1.0000 (Support: 30)
  - **Positive (2):** Precision = 1.0000 | Recall = 1.0000 | $F_1$-Score = 1.0000 (Support: 30)
- **Sample Inference Verification (6/6 Correct):**
  - Sample 1 (Pos): *"The engine performance is outstanding and accelerates smoothly."* $\rightarrow$ **Positive** (P(Pos)=0.999)
  - Sample 2 (Pos): *"Infotainment screen is impressive, responsive, and intuitive to use."* $\rightarrow$ **Positive** (P(Pos)=0.986)
  - Sample 3 (Neg): *"The engine performance is terrible and overheats quickly."* $\rightarrow$ **Negative** (P(Neg)=0.998)
  - Sample 4 (Neg): *"Cabin noise level is dreadful and intrusive during travel."* $\rightarrow$ **Negative** (P(Neg)=0.999)
  - Sample 5 (Neu): *"The engine displacement is measured at 2.0 liters."* $\rightarrow$ **Neutral** (P(Neu)=0.816)
  - Sample 6 (Neu): *"Electrical battery system operates at 40 volts."* $\rightarrow$ **Neutral** (P(Neu)=0.998)

## 5. Limitations of Synthetic Data & Scope
- Training an LSTM model from scratch on small synthetic text corpora (360 training samples) without pre-trained word embeddings (e.g., GloVe / Word2Vec) allows fast convergence on domain-specific vocabulary.
- Out-of-vocabulary (OOV) terms in real-world messy text require larger pre-trained embeddings or sub-word tokenizers (e.g. Byte-Pair Encoding).

## 6. How to Run
1. Navigate to `Assignments/Assignment_08/`:
   ```bash
   cd Assignments/Assignment_08
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_08.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).
