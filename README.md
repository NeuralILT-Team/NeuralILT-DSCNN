# Accelerating Inverse Lithography Technology using Depthwise Separable CNNs

CMPE 257 – Spring 2026  
Team: Pooja Singh, Rana Shamoun, Rishi Sheth, Pramod Yadav  

---

## 📌 Overview

This project explores the use of deep learning to accelerate **Inverse Lithography Technology (ILT)**.

We compare:

- Baseline U-Net CNN  
- Depthwise Separable CNN (DS-CNN)  

The goal is to improve computational efficiency while maintaining mask prediction quality.

---

## 📂 Dataset

We use the **LithoBench MetalSet dataset**.

Required structure:

data/raw/MetalSet/
  target/
  litho/

---

## ⚙️ Quick Start

Run preprocessing:

python -m src.data.preprocess

Run training:

python -m src.train

---

## 📊 Evaluation

Models are evaluated using:

- MSE (Mean Squared Error)  
- SSIM (Structural Similarity Index)  
- EPE (Edge Placement Error)  

---

## 🐳 Docker (Optional)

Build:

docker build -t ilt-project .

Run:

docker run -it -v $(pwd):/app ilt-project

---

## 📌 Notes

- Use subsets of data for local runs if needed  
- Full dataset experiments may require high-memory or GPU systems  

---

## 🚀 Project Goal

Improve efficiency of ILT models while maintaining high-quality mask predictions.
