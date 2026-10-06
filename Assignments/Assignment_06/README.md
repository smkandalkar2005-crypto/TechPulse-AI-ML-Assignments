# Assignment 6 — Traffic Sign Classification using CNN and GTSRB

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Train a **Convolutional Neural Network (CNN)** to classify German Traffic Sign Recognition Benchmark (**GTSRB**) images across 43 distinct traffic sign categories.

## 2. GTSRB Dataset & Preprocessing Description
- **Dataset:** Official German Traffic Sign Recognition Benchmark (GTSRB) downloaded automatically via `torchvision.datasets.GTSRB`.
- **Classes:** 43 distinct traffic sign categories (Speed limits, Stop, Yield, No Entry, Warnings, Directions, etc.).
- **Execution Mode:** Practical balanced subset sampler (`USE_SUBSET = True`) to ensure quick execution on standard student hardware.
  - **Train Subset:** 100 samples per class = 4,300 images total.
  - **Test Subset:** 30 samples per class = 1,290 images total.
- **Image Specifications:** RGB color images resized to $32 \times 32$ pixels, pixel values normalized to $[0, 1]$.

## 3. CNN Architecture
- **Conv Block 1:** `Conv2D(3, 32, 3x3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)`
- **Conv Block 2:** `Conv2D(32, 64, 3x3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)`
- **Fully Connected:** `Flatten` $\rightarrow$ `Linear(64*8*8, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(128, 43)`
- **Optimizer & Loss:** Adam (`lr=0.001`), `nn.CrossEntropyLoss()`.
- **Epochs:** 8 training epochs.

## 4. Actual Executed Performance Summary
All values below come directly from the executed notebook (`TTL_Assignment_06.ipynb`):

- **Final Test Accuracy:** **68.29%** (881 / 1,290 correct test predictions).
- **Training Loss:** Decreased from **3.6361** (Epoch 1) to **0.9014** (Epoch 8).
- **Training Accuracy:** Increased from **5.74%** (Epoch 1) to **71.07%** (Epoch 8).
- **Test Loss:** Decreased from **3.2881** (Epoch 1) to **1.2242** (Epoch 8).
- **High-Performing Classes (F1-Score > 0.90):**
  - No passing veh over 3.5t: **0.9474**
  - Veh > 3.5t prohibited: **0.9492**
  - Go straight or right: **0.9333**
  - Go straight or left: **0.9286**
  - Yield: **0.9206**
  - Bumpy road: **0.9206**
  - Priority road: **0.9153**
  - Stop: **0.9091**

## 5. Key Evaluation Outputs Included
1. Sample grid of 16 raw GTSRB training images with class index and label text.
2. Loss and Accuracy convergence curves across 8 epochs.
3. Full 43-class classification report (Precision, Recall, F1-Score, Support).
4. $43 \times 43$ Confusion Matrix heatmap.
5. Grid of 15 sample test images showing True vs Predicted labels with color-coded correctness (green = correct, red = misclassified).

## 6. How to Run
1. Open terminal and navigate to `Assignments/Assignment_06/`:
   ```bash
   cd Assignments/Assignment_06
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_06.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).

## 7. Execution Status
- **Executed successfully without errors.**
- **Notebook Output:** `TTL_Assignment_06.ipynb` (444 KB executed notebook).
