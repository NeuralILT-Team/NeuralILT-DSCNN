
# Accelerating Inverse Lithography Technology using Depthwise Separable CNNs

CMPE 257 – Spring 2026  
Team: Pooja Singh, Rana Shamoun, Rishi Sheth, Pramod Yadav  

---

## 📌 Project Overview

This project focuses on accelerating **Inverse Lithography Technology (ILT)** using deep learning.

We implement and compare:

- **Baseline Model**: U-Net style CNN (vanilla architecture)
- **Proposed Model**: Depthwise Separable CNN (DS-CNN)

The goal is to evaluate whether lightweight CNN architectures can reduce computational cost while maintaining mask prediction quality.

---

## 👩‍💻 My Contribution

- Designed and structured the GitHub repository
- Implemented the **data preprocessing pipeline**
- Handled **dataset integration (LithoBench MetalSet)**
- Implemented **baseline model pipeline setup**
- Enabled **scalable preprocessing using MAX_SAMPLES**

---

## 📂 Dataset

We use the **LithoBench MetalSet dataset**, where:

- `target/` → input layout patterns  
- `litho/` → ground truth mask outputs  

### Required Structure:
data/raw/MetalSet/
├── target/
├── litho/


---

## ⚙️ Preprocessing

The preprocessing script:

- Converts images to grayscale
- Normalizes pixel values to [0, 1]
- Saves processed layout-mask pairs

### ▶️ Run preprocessing

```bash
python -m src.data.preprocess