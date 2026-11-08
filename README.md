# Handwritten Character Recognition (MNIST) with CNNs

An end-to-end Machine Learning and Deep Learning pipeline that classifies handwritten digits (0–9) using Convolutional Neural Networks (CNNs).

This project strictly adheres to an **empirical engineering approach**:
1. Starting from a fast **non-neural baseline** (Logistic Regression).
2. Building a minimal **baseline CNN** (SimpleCNN) to capture 2D spatial features.
3. Systematically testing an **enhanced deep CNN** (DeepCNN with Batch Normalization & Dropout) to test what extra depth and regularization actually buy.
4. Measuring **inference latency** (ms per sample) and **training curves** (diagnosing overfitting).
5. Performing a rigorous **error analysis on the 20 worst predictions** (most confident mistakes).

---

## 📊 Summary of Experimental Results

| Model | Parameters | Training Time | Test Accuracy (%) | Inference Latency (ms/sample) | Key Takeaway / What the Change Bought |
|---|---|---|---|---|---|
| **Logistic Regression** (Non-Neural Baseline) | 7,850 | 20.36 s | **90.43%** | **0.0178 ms** | Fast linear benchmark; treats pixels independently without spatial awareness. |
| **SimpleCNN** (1 Conv + MaxPool + FC) | 27,210 | 295.26 s | **97.93%** | **0.4928 ms** | **+7.50% accuracy gain** by exploiting local 2D spatial relationships with a 3×3 convolution. |
| **DeepCNN** (2 Conv + BN + Dropout + FC) | 421,834 | 691.12 s | **99.03%** | **2.3008 ms** | **+1.10% incremental gain**, but 15.5× more parameters and 4.6× slower inference. |

> **Key Finding:** Moving from flat pixels (Logistic Regression) to a minimal 1-layer CNN bought the largest accuracy leap (**+7.50%**). Adding deeper layers, batch normalization, and dropout further boosted accuracy to over **99%**, but at the cost of higher compute and latency.

---

## 🏗️ Model Architectures

### 1. Non-Neural Baseline: Logistic Regression
- **Input:** Flattened 784-dimensional vector ($28 \times 28$).
- **Solver:** L-BFGS, Multinomial Cross-Entropy.
- **Purpose:** Establish the minimum performance floor that any neural network must comfortably exceed.

### 2. SimpleCNN (Baseline Neural Network)
- `Conv2d(1, 16, kernel_size=3)` $\rightarrow$ Output: $16 \times 26 \times 26$
- `ReLU`
- `MaxPool2d(2, 2)` $\rightarrow$ Output: $16 \times 13 \times 13$
- `Flatten` $\rightarrow$ $2,704$ features
- `Linear(2704, 10)`

### 3. DeepCNN (Regularized Deep Network)
- **Block 1:** `Conv2d(1, 32, 3x3, pad=1)` + `BatchNorm2d(32)` + `ReLU` + `MaxPool2d(2, 2)`
- **Block 2:** `Conv2d(32, 64, 3x3, pad=1)` + `BatchNorm2d(64)` + `ReLU` + `MaxPool2d(2, 2)` + `Dropout(0.25)`
- **Dense Head:** `Linear(3136, 128)` + `ReLU` + `Dropout(0.5)` + `Linear(128, 10)`

---

## 📈 Training Curves & Overfitting Analysis

The training and validation curves are saved to [`outputs/training_curves.png`](outputs/training_curves.png).

- **SimpleCNN:** Rapidly converges within 3 epochs. By epoch 5, training loss continues to drop ($0.046$) while validation loss begins slightly plateauing/ticking up ($0.077$), indicating the onset of mild overfitting due to lack of dropout regularization.
- **DeepCNN:** Batch Normalization stabilizes gradient flow, while Dropout ($p=0.25$ and $p=0.5$) keeps validation loss decreasing steadily in tandem with training loss, suppressing overfitting.

---

## 🔍 Error Analysis: 20 Worst Predictions

Even at >99% test accuracy, a model makes errors. The critical engineering task is understanding **why** it failed.

We specifically extract the **20 test images where the model was most confidently wrong** (highest predicted probability for an incorrect class). The visual grid is saved at [`outputs/worst_20_errors.png`](outputs/worst_20_errors.png) and documented in [`outputs/error_analysis.md`](outputs/error_analysis.md).

### Primary Failure Modes Identified:
1. **Ambiguous Digit Closures (4 vs 9):** Digits where the top vertical stems touch or nearly close mimic a loop, confusing `4` with `9`.
2. **Missing Strokes / Slanted Stems (1 vs 7):** European-style crossed 7s with faint crosses, or 1s drawn with a top serif resembling a 7.
3. **Curvature Ambiguity (3 vs 5 and 3 vs 8):** Faint loops or cursive handwriting where open strokes meet the midline mimic closed loops of an 8 or upper bars of a 5.
4. **Human Labeling Artifacts:** A small number of ground truth labels in MNIST are debatable even to a human eye due to extreme pen stroke deformation.

---

## 🚀 How to Run

### 1. Setup & Installation
```bash
git clone https://github.com/your-username/handwritten-character-recognition.git
cd handwritten-character-recognition
pip install -r requirements.txt
```

### 2. Run Entire Pipeline (Download + Baseline + CNN Training + Error Analysis)
```bash
python main.py
```
Outputs generated in `outputs/`:
- `SimpleCNN.pth` (Baseline weights)
- `DeepCNN.pth` (Deep CNN weights)
- `training_curves.png` (Train vs Val Loss/Accuracy comparison)
- `worst_20_errors.png` (Grid of the 20 worst misclassifications)
- `error_analysis.md` (Detailed tabular breakdown of failure reasons)
- `benchmark_summary.txt` (Ablation and performance table)

### 3. Test on an Individual MNIST Sample
```bash
# Test sample index #42 from the MNIST test set
python predict_sample.py 42
```
Outputs the actual digit, predicted digit, model confidence, and an ASCII probability bar chart.

---

## 🌟 EMNIST ByClass Upgrade (Full 62-Class Recognition)

To support full alphanumeric recognition beyond digits, the project was upgraded to the **EMNIST ByClass** benchmark (62 classes).

### Class Mapping (62 Classes)
- **Indices 0–9:** Digits `'0'` to `'9'`
- **Indices 10–35:** Uppercase letters `'A'` to `'Z'`
- **Indices 36–61:** Lowercase letters `'a'` to `'z'`

### EMNIST Performance Summary
- **Model:** `DeepCNN(num_classes=62)` (saved as `outputs/EMNIST_DeepCNN.pth`)
- **Overall Test Accuracy:** **83.32%** on official EMNIST test split
- **Macro Precision:** 69.33%
- **Macro Recall:** 65.22%
- **Macro F1-Score:** 64.45%
- **Orientation Handling:** NIST raw images have swapped row/column axes; corrected using `transpose(Image.Transpose.TRANSPOSE)` in the data pipeline.

> **Viva Insight on Case Ambiguity:**
> In handwritten text, glyphs like `C` vs `c`, `O` vs `o`, `S` vs `s`, `V` vs `v`, `W` vs `w`, `X` vs `x`, and `Z` vs `z` have virtually identical morphology when presented as isolated 28x28 crops without relative baseline/cap-height scale. The model typically predicts the uppercase variant. This is why EMNIST provides both ByClass (62 classes) and ByMerge (47 classes).

### How to Run EMNIST Training & Predictions
1. **Train EMNIST Model:**
   ```bash
   python train_emnist.py
   ```
2. **Predict on Any Custom Image (JPG/PNG/BMP):**
   ```bash
   # Predict on any handwritten image
   python predict_image.py path/to/your_drawing.png
   ```

---

## 🖥️ Professional Desktop GUI Application (`desktop_gui.py`)

A modern desktop application built with **PySide6 (Qt)** and **PyTorch**, featuring a warm editorial palette (`#F0EFEC`, `#FFFFFF`, `#DED8CF`, `#B9AA98`, `#897765`, `#39342F`) and dedicated mouse-driven drawing.

### 🎨 Key Features:
1. **Dedicated Mouse Drawing Canvas:**
   - **Pure Mouse Input:** Left-click to start, drag to draw smooth anti-aliased strokes, release to finish.
   - 360×360 canvas with crisp warm border (`#DDD7CF`) on pure white (`#FFFFFF`).
2. **Interactive Controls:**
   - **Undo (`Ctrl+Z`) & Redo (`Ctrl+Y`)** stack management.
   - **Adjustable Pen Thickness:** Clean slider from 6 px to 36 px.
   - **Clear Canvas (`Escape` or `Ctrl+Delete`)** with instant reset.
   - **Export Drawing (`Ctrl+S`):** Save high-res drawing as PNG.
3. **Multiple Handwritten Character Recognition & OpenCV Segmentation:**
   - Recognizes words (`HELLO`), multiple letters (`AB`, `ABC`), digits (`123`), and mixed sequences (`A1B`).
   - Connected component analysis with stroke grouping (handles dots on 'i'/'j' and crossbars).
   - Vertical projection profiling to split touching cursive characters.
   - Automatic left-to-right sorting and aspect-ratio preserving centering to 28×28 EMNIST tensors.
4. **Dual Recognition Workflow:**
   - **Recognize Text (`Enter` / `Return`):** Primary brown button (`#897765`). Automatically segments multiple characters and outputs combined string, confidence, and character breakdown.
   - **Recognize Single (`Shift+Enter`):** Dedicated button for single character classification with Top-5 candidate probabilities.
5. **Real-Time Visual Feedback:**
   - Dynamic hero card showing recognized string (with dynamic auto-scaled typography).
   - Processed preview box displaying either 28×28 CNN input or segmented canvas with numbered bounding boxes.
   - Character breakdown with progress bars and classification categories.
6. **Session Recognition History:**
   - Clean data table logging recent drawings with thumbnails, predicted text/characters, categories, confidence scores, and timestamps.
7. **Non-Blocking Architecture:**
   - Model loaded once at startup. Inference runs asynchronously in Qt background worker threads (`SequenceInferenceWorker`, `InferenceWorker`) to guarantee zero UI stuttering.

### 🚀 How to Run the Desktop GUI:
```bash
python desktop_gui.py
```

### ⌨️ Keyboard Shortcuts:
| Action | Shortcut |
|---|---|
| **Recognize Text (Sequence)** | `Enter` / `Return` |
| **Recognize Single Character** | `Shift+Enter` |
| **Clear Canvas** | `Escape` or `Ctrl+Delete` |
| **Undo Stroke** | `Ctrl+Z` |
| **Redo Stroke** | `Ctrl+Y` or `Ctrl+Shift+Z` |
| **Save Drawing** | `Ctrl+S` |

### 🧪 Automated Testing:
```bash
# Run full GUI test suite (8 tests)
python test_gui.py

# Run multi-character sequence validation (AB, ABC, 123, A1B, HELLO)
python test_multi_character.py
```

### 📦 Packaging as Windows Executable (.exe) with PyInstaller:
To bundle the application into a standalone Windows `.exe`:
```bash
pip install pyinstaller
pyinstaller --name="HandwritingAI" --windowed --add-data="outputs/EMNIST_DeepCNN.pth;outputs" --icon=NONE desktop_gui.py
```
The resulting executable will be placed in the `dist/HandwritingAI/` folder.
