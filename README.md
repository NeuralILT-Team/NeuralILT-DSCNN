# Accelerating Inverse Lithography Technology using Depthwise Separable CNNs

CMPE 257 – Spring 2026  
Team: Pooja Singh, Rana Shamoun, Rishi Sheth, Pramod Yadav  

---

## 📌 Project Overview

This project focuses on accelerating Inverse Lithography Technology (ILT) using deep learning.

We implement and compare:

- Baseline Model: U-Net style CNN (vanilla architecture)
- Proposed Model: Depthwise Separable CNN (DS-CNN)

The goal is to evaluate whether lightweight CNN architectures can reduce computational cost while maintaining mask prediction quality.

---

## 👩‍💻 My Contribution

- Designed and structured the GitHub repository  
- Implemented the data preprocessing pipeline  
- Handled dataset integration (LithoBench MetalSet)  
- Set up baseline training pipeline (vanilla CNN / U-Net)  
- Enabled scalable preprocessing using MAX_SAMPLES  
- Added Docker support for reproducibility  
- Created a docker-compose.yml (currently empty) for potential future multi-service setup  

---

## 📂 Dataset

We use the LithoBench MetalSet dataset:

- target → input layout patterns  
- litho → ground truth mask outputs  

Required structure:

data/raw/MetalSet/
  target/
  litho/

---

## ⚙️ Preprocessing

The preprocessing script:

- Converts images to grayscale  
- Normalizes pixel values to [0, 1]  
- Saves processed layout-mask pairs  

Run preprocessing (full dataset):

python -m src.data.preprocess

Run with sample (recommended for local machines):

MAX_SAMPLES=5000 python -m src.data.preprocess

Other examples:

MAX_SAMPLES=1000 python -m src.data.preprocess  
MAX_SAMPLES=3000 python -m src.data.preprocess  

---

## 🧠 How Sampling Works

- MAX_SAMPLES controls number of layout-mask pairs  
- If not set → full dataset is used  
- If set → subset is used  

---

## ⚠️ Important Notes

- Full preprocessing may require 70GB+ storage  
- Use subsets for local runs  
- Designed for scalable execution across environments  

---

## 🧠 Training (Baseline Model)

python -m src.train

---

## 📊 Evaluation Metrics

- MSE (Mean Squared Error)  
- SSIM (Structural Similarity Index)  
- EPE (Edge Placement Error)  

---

## 🐳 Docker Usage

Docker is used to ensure reproducibility and consistent environment setup.

Build Docker image:

docker build -t ilt-project .

Run container:

docker run -it -v $(pwd):/app ilt-project

Inside container:

python -m src.data.preprocess  
python -m src.train  

Run with subset inside Docker:

MAX_SAMPLES=5000 python -m src.data.preprocess  

---

## 🧩 docker-compose (Future Use)

A docker-compose.yml file has been added (currently empty) to support future extensions such as:

- Model serving API  
- Visualization dashboards (Streamlit/Gradio)  
- Multi-container workflows  

---

## 🗂️ Project Structure

ilt-ds-cnn/
  README.md  
  src/
    data/
      preprocess.py
      dataset.py
    train.py
  data/
    raw/
      MetalSet/
        target/
        litho/
    processed/
  configs/      (empty)
  scripts/      (empty)
  requirements.txt
  Dockerfile
  docker-compose.yml

---

## 🛠️ Files Created / Modified

- src/data/preprocess.py → preprocessing pipeline  
- src/data/dataset.py → dataset loader  
- README.md → documentation  
- requirements.txt → dependencies  
- Dockerfile → container setup  
- docker-compose.yml → created (currently empty)  
- configs/ → created (empty)  
- scripts/ → created (empty)  

---

## 🚀 Team Workflow

- Pooja: preprocessing + baseline  
- Rana:
- Pramod:
- Rishi:

---

## 💡 Notes

Due to hardware constraints:

- Local runs use dataset subsets  
- Full dataset runs should be done on high-memory or GPU systems  

---

## 📌 Summary

Raw Dataset → Preprocessing → Baseline Training → Evaluation → DS-CNN Comparison
