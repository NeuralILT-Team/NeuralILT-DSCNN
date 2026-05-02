# CMPE 257 Oral Exam Preparation — NeuralILT-DSCNN Project

## Critical Review + Question Bank + Learning Guide

---

## Part 1: Critical Review of the Project

### What the Project Does
The project compares a **Baseline U-Net** (standard 3×3 convolutions, 31M params) against a **DS-CNN U-Net** (depthwise separable convolutions, 6M params) for **Inverse Lithography Technology (ILT)** — predicting optimized photomask patterns from target semiconductor layouts using the LithoBench MetalSet (16,472 tiles).

### Actual Results (from [RESULTS.md](../RESULTS.md))

#### Accuracy Across Epochs

| Metric | Epoch | Baseline | DS-CNN | Difference |
|--------|-------|----------|--------|------------|
| **MSE** | 20 | 0.000098 | 0.000102 | +4.1% |
| | 70 | 0.000046 | 0.000068 | +47.8% |
| | 100 | 0.000058 | 0.000096 | +65.5% |
| **SSIM** | 20 | 0.9663 | 0.9649 | −0.14% |
| | 70 | 0.9823 | 0.9761 | −0.64% |
| | 100 | 0.9793 | 0.9673 | −1.23% |

#### Efficiency (Architecture-Level — Same at All Epochs)

| Metric | Baseline | DS-CNN | Reduction |
|--------|----------|--------|-----------|
| Params | 31.0M | 6.0M | **80.6%** |
| FLOPs | 109.3B | 28.5B | **73.9%** |
| Runtime | 19.04 ms | 3.78 ms | **80.1%** |

#### Wider DS-CNN Variant (1.5×)

| Metric | Standard DS-CNN | Wider 1.5× | Baseline |
|--------|----------------|------------|----------|
| Params | 6.0M | 13.4M | 31.0M |
| FLOPs | 28.5B | 63.3B | 109.3B |
| SSIM (20 ep) | 0.9657 | 0.9714 | 0.9663 |
| Runtime | 3.78 ms | 30.02 ms | 19.04 ms |

### Strengths ✅
1. **Clean experimental design** — same architecture depth `[64,128,256,512]`, same optimizer, same data split, same seed — isolates the effect of convolution type
2. **Comprehensive metrics** — MSE (pixel), SSIM (structural), EPE (domain-specific edge placement), FLOPs, params, runtime, memory
3. **Reproducibility** — fixed seed 42, YAML configs, checkpoint resume, split JSON
4. **Generalization test** — StdMetal (271 unseen tiles) tests out-of-distribution performance
5. **Hyperparameter sweep** — LR sweep across 5 values, width multiplier sweep across 3 values
6. **Strong efficiency gains** — 81% fewer params, 74% fewer FLOPs, 5× faster inference
7. **Multi-epoch comparison** — results at 20, 70, and 100 epochs show training dynamics and optimal stopping point

### Weaknesses / Areas for Critique ⚠️
1. **MSE as loss function is suboptimal** — SSIM loss or a combined loss would better optimize for structural fidelity (the metric they care about)
2. **EPE = 0.0000 for baseline at epochs 20 and 70 is suspicious** — likely means all predicted edges exactly overlap GT edges at the binarization threshold, which could indicate the threshold (0.5) is too coarse or the metric implementation has edge cases
3. **Wider DS-CNN (1.5×) is slower than baseline** — 30.02 ms vs 19.04 ms, undermining the efficiency argument; the standard DS-CNN at 3.78 ms is the real efficiency win
4. **No cross-validation** — single 80/10/10 split; results could be split-dependent
5. **No statistical significance testing** — no confidence intervals or multiple runs
6. **Generalization drop is significant** — SSIM drops from 0.971 → 0.893 on StdMetal, suggesting overfitting to MetalSet patterns
7. **No residual connections in DS blocks** — MobileNetV2's inverted residuals would likely improve DS-CNN accuracy
8. **Cosine scheduler with T_max=100** — best checkpoint occurs at epoch 66-70, not at epoch 100; the LR schedule's tail end doesn't help
9. **100-epoch results show regression** — both models' test metrics worsen from epoch 70 → 100, indicating the `best_model.pt` from the 100-epoch run may not be the global best

---

## Part 2: Oral Exam Questions (Organized by CMPE 257 Topic)

### Category 1: Machine Learning Concepts & Feasibility

**Q1. What type of machine learning problem is this project solving? Classify it by learning type.**

> **Answer**: This is a **supervised learning** problem — specifically **image-to-image regression**. The model learns a mapping from input layout images (X) to output lithography mask images (Y) using paired training data. It's regression (not classification) because the output is continuous pixel values in [0,1], not discrete labels. The sigmoid activation in [`baseline_unet.py`](../src/models/baseline_unet.py) constrains outputs to [0,1].

**Q2. Why is this problem feasible for machine learning? What makes it learnable?**

> **Answer**: Feasibility rests on three pillars: (1) **sufficient data** — 16,472 paired tiles from LithoBench, split 80/10/10; (2) **a learnable mapping exists** — the physics of optical lithography creates a deterministic (but complex) relationship between layouts and optimal masks; (3) **the target function has structure** — spatial locality (nearby pixels are correlated) makes CNNs appropriate. The low final MSE (~0.00005) confirms the function is learnable.

**Q3. Is this project solving a well-posed problem? What would make it ill-posed?**

> **Answer**: It's well-posed because each layout has a unique optimal mask (the physics is deterministic). It would become ill-posed if multiple valid masks existed for the same layout (non-unique solutions), or if small layout changes caused large mask changes (instability). The smooth MSE loss landscape and stable training curves confirm well-posedness.

---

### Category 2: Theory of Generalization

**Q4. How does the project test generalization? What did the results show?**

> **Answer**: The project tests generalization in two ways: (1) **in-distribution** — the 10% test split of MetalSet (1,643 tiles never seen during training); (2) **out-of-distribution** — StdMetal (271 tiles from a different layout distribution, never seen during training). Results show SSIM drops from 0.971 → 0.893 on StdMetal, indicating the model has learned MetalSet-specific patterns that don't fully transfer. This is a ~8% generalization gap.

**Q5. According to VC theory, which model should generalize better — the 31M-parameter baseline or the 6M-parameter DS-CNN? Did your results confirm this?**

> **Answer**: VC theory predicts that models with **fewer effective parameters** (lower VC dimension) should generalize better, assuming both achieve similar training error. The DS-CNN has 5× fewer parameters, so VC theory predicts it should generalize better. However, the results are **mixed** — on MetalSet test, the baseline actually achieves better SSIM (0.982 vs 0.976 at epoch 70). The consistency test at 100 epochs shows both models produce similar predictions on unseen layouts (SSIM > 0.86). This highlights that VC bounds are **loose** — they give worst-case guarantees, not tight predictions.

**Q6. What is the generalization bound, and how does model complexity relate to it in this project?**

> **Answer**: The generalization bound states: `E_out ≤ E_in + Ω(d_VC, N)` where Ω grows with VC dimension and shrinks with sample size N. With N=13,178 training samples and d_VC proportional to parameters (31M vs 6M), the DS-CNN has a tighter bound. But the bound is vacuous for neural networks (Ω >> 1 for millions of parameters). In practice, **regularization** (BatchNorm, early stopping, data augmentation) matters more than raw parameter count for generalization.

**Q7. The project uses a fixed 80/10/10 split with seed=42. What are the risks of this approach?**

> **Answer**: Risks include: (1) **split-dependent results** — a different seed could yield different metrics; (2) **no confidence intervals** — we can't assess result variance; (3) **potential data leakage** — if preprocessing shuffles before splitting, the split re-shuffles with seed=42, which is correct but the preprocessing shuffle is unseeded. Better approaches: k-fold cross-validation, or multiple random splits with averaged results.

---

### Category 3: Bias and Variance

**Q8. Analyze the bias-variance tradeoff between the baseline and DS-CNN models.**

> **Answer**: 
> - **Baseline (31M params)**: Lower bias (more expressive, can fit complex patterns), potentially higher variance (more parameters to overfit). Evidence: achieves lower training loss (0.000025) and lower val loss (0.000046).
> - **DS-CNN (6M params)**: Higher bias (depthwise separable convolutions restrict the hypothesis space — each channel is filtered independently before mixing), lower variance. Evidence: training loss (0.000042) is higher, but the train-val gap is smaller (0.000042 vs 0.000068 = 62% gap, compared to baseline's 0.000025 vs 0.000046 = 84% gap).
> - The DS-CNN's **higher bias** manifests as the SSIM gap (0.976 vs 0.982). The **lower variance** manifests as more stable validation curves in later epochs.

**Q9. Is the baseline model overfitting? How can you tell from the training logs?**

> **Answer**: Looking at the training logs in [RESULTS.md](../RESULTS.md), the baseline shows signs of **mild overfitting**: at epoch 68, val_loss spikes to 0.000059 and SSIM drops to 0.9597, while training loss remains stable at 0.000025. This train-val divergence indicates the model is memorizing training patterns that don't generalize to validation. At 100 epochs, the test MSE increases from 0.000046 → 0.000058, further confirming overfitting. The DS-CNN shows more stable validation metrics (0.000068-0.000069 range), suggesting less overfitting — consistent with its lower model complexity.

**Q10. How does data augmentation in this project relate to bias-variance?**

> **Answer**: The augmentation (horizontal flip, vertical flip, 90° rotation) effectively increases the training set size by 8× (2 × 2 × 4 orientations). This **reduces variance** without changing bias — the model sees more diverse examples, making it harder to memorize specific patterns. Critically, only **geometric** transforms are used (no brightness/contrast changes) because pixel values have physical meaning in lithography — this is a domain-informed design decision that avoids introducing bias through invalid augmentations.

---

### Category 4: Linear Models, Nonlinear Transformation

**Q11. Why can't a linear model solve this problem? What nonlinear transformations does the U-Net apply?**

> **Answer**: A linear model (Y = WX + b) can only learn linear pixel-to-pixel mappings. ILT mask optimization involves **nonlinear optical proximity effects** — a pixel's optimal mask value depends on its spatial neighborhood in complex, nonlinear ways. The U-Net applies multiple nonlinear transformations: (1) **ReLU activations** after every convolution (in [`blocks.py`](../src/models/blocks.py)); (2) **sigmoid** output activation (in [`baseline_unet.py`](../src/models/baseline_unet.py)); (3) **hierarchical feature extraction** through the encoder-decoder structure captures multi-scale spatial relationships.

**Q12. What is the role of the encoder-decoder (U-Net) architecture as a nonlinear feature transformation?**

> **Answer**: The encoder progressively downsamples (MaxPool2d) and increases channels `[1→64→128→256→512→1024]`, creating increasingly abstract, spatially-compressed representations. This is a **learned nonlinear feature transformation** — analogous to applying Φ(x) in kernel methods, but learned end-to-end. The decoder upsamples back to the original resolution. Skip connections concatenate encoder features with decoder features, preserving fine spatial details that would otherwise be lost — this is critical for pixel-precise mask prediction.

---

### Category 5: Regularization and Validation

**Q13. What regularization techniques are used in this project? Evaluate their effectiveness.**

> **Answer**: 
> 1. **BatchNorm** ([`blocks.py`](../src/models/blocks.py)) — after every conv layer; acts as implicit regularization by adding noise through mini-batch statistics
> 2. **Gradient clipping** (grad_clip=1.0 in config) — prevents exploding gradients, stabilizes training
> 3. **Data augmentation** — 8× effective dataset expansion via geometric transforms
> 4. **Early stopping** (implicit) — best model checkpoint saved based on validation loss
> 5. **Weight decay = 0.0** — notably, L2 regularization is **not used**, which is a missed opportunity
> 6. **Cosine annealing LR** — gradually reduces learning rate, acting as implicit regularization in later epochs
> 
> Missing: **Dropout** (not used anywhere), **L1/L2 weight decay** (set to 0). Adding weight_decay=1e-4 could reduce the baseline's overfitting observed at epoch 100.

**Q14. Explain the validation strategy. Is it sufficient for a master's-level project?**

> **Answer**: The project uses a **holdout validation** strategy: 80% train, 10% validation, 10% test, with fixed seed=42. Validation loss is monitored every epoch to save the best checkpoint. This is **minimally sufficient** but has weaknesses: (1) no k-fold cross-validation means results depend on the specific split; (2) no repeated experiments means no confidence intervals; (3) the test set is only used once at the end (correct practice). For a master's project, adding 5-fold CV or at least 3 random seeds would strengthen the conclusions.

**Q15. The cosine scheduler is configured with T_max=100. The best checkpoint occurs at epoch 66-70. What's the impact of training to 100?**

> **Answer**: The cosine annealing scheduler in [`train.py`](../src/train.py) uses `T_max=100` from config. At epoch 70/100, the LR has decayed to ~0.000001 (near the minimum). Training to 100 epochs shows **diminishing returns** — both models' test metrics actually worsen (baseline MSE: 0.000046 → 0.000058, DS-CNN MSE: 0.000068 → 0.000096). This suggests the very-low-LR tail of cosine annealing doesn't help and may cause the model to drift from its optimal point. The `best_model.pt` checkpoint mechanism correctly captures the epoch-70 optimum.

---

### Category 6: Kernel Methods and Convolutions

**Q16. How do depthwise separable convolutions relate to kernel methods? Explain the mathematical connection.**

> **Answer**: A standard 3×3 convolution applies a **joint spatial-channel kernel**: for C_in input channels and C_out output channels, it learns C_out kernels of size C_in × 3 × 3, costing `H×W×C_in×C_out×9` FLOPs. A depthwise separable convolution **factorizes** this into two steps:
> 1. **Depthwise** (spatial kernel): C_in separate 3×3 kernels, one per channel → `H×W×C_in×9` FLOPs
> 2. **Pointwise** (channel kernel): C_out 1×1 kernels mixing channels → `H×W×C_in×C_out` FLOPs
> 
> Total: `H×W×C_in×(9 + C_out)` vs `H×W×C_in×C_out×9`. For C_out=64: ratio = (9+64)/(64×9) ≈ 1/8.
> 
> This is analogous to **kernel decomposition** in kernel methods — instead of computing the full kernel matrix K(x,x'), you approximate it as a product of simpler kernels. The depthwise conv is a **separable kernel** (spatial × channel), trading expressiveness for efficiency.

**Q17. The depthwise conv uses `groups=in_ch`. Explain what grouped convolutions are and why this matters.**

> **Answer**: In [`blocks.py`](../src/models/blocks.py), `groups=in_ch` means each input channel gets its own independent 3×3 filter — no cross-channel interaction during spatial filtering. This is the extreme case of grouped convolutions (groups=1 is standard, groups=in_ch is fully depthwise). It matters because: (1) it **restricts the hypothesis space** — the model can't learn joint spatial-channel features in a single operation; (2) it **reduces parameters** from `C_in×C_out×K²` to `C_in×K²`; (3) the subsequent 1×1 pointwise conv restores channel mixing. This factorization assumes spatial and channel features are **approximately separable** — a strong assumption that works well in practice (MobileNet proved this).

---

### Category 7: Neural Networks (Deep Learning)

**Q18. Explain the U-Net architecture used in this project. Why is it appropriate for this task?**

> **Answer**: The U-Net (from [`baseline_unet.py`](../src/models/baseline_unet.py)) is an encoder-decoder with skip connections:

```
Encoder: 1→64→128→256→512 (MaxPool between levels)
Bottleneck: 512→1024
Decoder: 1024→512→256→128→64 (ConvTranspose2d upsampling)
Output: 64→1 (1×1 conv + sigmoid)
```

> Skip connections concatenate encoder features with decoder features at each level. This is appropriate for ILT because: (1) **pixel-precise output** is needed (skip connections preserve spatial detail); (2) **multi-scale context** matters (the encoder captures both local edge features and global layout structure); (3) **same input/output resolution** (256×256 → 256×256).

**Q19. Walk through the forward pass of the DS-CNN model. What happens at each stage?**

> **Answer**: From [`ds_unet.py`](../src/models/ds_unet.py):
> 1. **Input**: [B, 1, 256, 256] grayscale layout
> 2. **Encoder Level 1**: DoubleConvDS(1→64) → MaxPool → [B, 64, 128, 128]
>    - Each DoubleConvDS = two DSConvBlocks: depthwise 3×3 → BN → ReLU → pointwise 1×1 → BN → ReLU
> 3. **Encoder Level 2**: DoubleConvDS(64→128) → MaxPool → [B, 128, 64, 64]
> 4. **Encoder Level 3**: DoubleConvDS(128→256) → MaxPool → [B, 256, 32, 32]
> 5. **Encoder Level 4**: DoubleConvDS(256→512) → MaxPool → [B, 512, 16, 16]
> 6. **Bottleneck**: DoubleConvDS(512→1024) → [B, 1024, 16, 16]
> 7. **Decoder**: ConvTranspose2d upsamples, concatenates with skip, DoubleConvDS reduces channels
> 8. **Output**: Conv2d(64→1) + sigmoid → [B, 1, 256, 256] predicted mask

**Q20. Why does the project use BatchNorm after every convolution? What would happen without it?**

> **Answer**: BatchNorm normalizes activations to zero mean and unit variance within each mini-batch. Benefits: (1) **stabilizes training** — prevents internal covariate shift as parameters update; (2) **enables higher learning rates** — normalized activations prevent gradient explosion; (3) **implicit regularization** — batch statistics add noise. Without BatchNorm, the DS-CNN would likely be harder to train because depthwise convolutions produce per-channel features with potentially very different scales. The project uses `bias=False` in convolutions because BatchNorm's learnable shift parameter (β) subsumes the bias.

**Q21. The project uses mixed precision training (float16). Explain why and what risks it introduces.**

> **Answer**: Mixed precision (enabled in [`train.py`](../src/train.py)) uses float16 for forward/backward passes and float32 for parameter updates. Benefits: (1) **halves GPU memory** — critical for the 12GB SJSU HPC GPU; (2) **faster computation** — Tensor Cores accelerate float16 ops. Risks: (1) **gradient underflow** — small gradients round to zero in float16; mitigated by `GradScaler` which scales loss up before backward, then unscales gradients; (2) **reduced precision** — metrics are computed in float32 for accuracy; (3) **numerical instability** — BatchNorm statistics can be noisy in float16.

**Q22. Explain gradient accumulation as used in this project. What problem does it solve?**

> **Answer**: Gradient accumulation processes `accum_steps=4` mini-batches before calling `optimizer.step()`. With batch_size=4 and accum_steps=4, the effective batch size is 16. This solves the **GPU memory constraint** — a batch of 16 images at 256×256 with a 31M-parameter model would OOM on 12GB. By accumulating gradients over 4 smaller batches, we get the same gradient estimate without the memory cost. The loss is divided by `accum_steps` to keep the gradient magnitude correct.

**Q23. Why does the project use Adam optimizer instead of SGD? What are the tradeoffs?**

> **Answer**: Adam combines momentum and adaptive learning rates per parameter. For this project: (1) **Adam converges faster** — important when GPU time is limited on shared HPC; (2) **less sensitive to LR choice** — the LR sweep shows LR=1e-3 works well, but even 1e-4 converges (just slower); (3) **handles sparse gradients** — useful when many mask pixels are zero. Tradeoffs: SGD with momentum often **generalizes better** (flatter minima) and uses less memory (no per-parameter state). The project could potentially improve generalization by switching to SGD with warmup.

---

### Category 8: Loss Functions and Optimization

**Q24. Why is MSE used as the loss function? What are its limitations for this task?**

> **Answer**: MSE measures average squared pixel difference: `L = (1/N) Σ(y_true - y_pred)²`. It's used because: (1) it's simple and differentiable; (2) it directly optimizes pixel accuracy. **Limitations**: (1) MSE treats all pixels equally — edge pixels (critical for lithography) get the same weight as interior pixels; (2) MSE doesn't capture **structural similarity** — two images with the same MSE can look very different; (3) MSE penalizes large errors quadratically, which can cause the model to produce "blurry" predictions that minimize worst-case error. A better choice would be **MSE + SSIM loss** or **focal loss** that emphasizes edge regions.

**Q25. The project measures SSIM but doesn't use it as a loss. Why might adding SSIM loss improve results?**

> **Answer**: SSIM measures structural similarity considering luminance, contrast, and structure in local windows. Using it as a loss would directly optimize for the metric the project cares about. The current setup optimizes MSE but evaluates SSIM — there's a **metric-loss mismatch**. Adding `L = α·MSE + (1-α)·(1-SSIM)` would: (1) preserve edge structure better; (2) potentially close the SSIM gap between baseline and DS-CNN; (3) produce sharper predictions. The code already imports SSIM computation ([`ssim.py`](../src/metrics/ssim.py)) — it just needs to be made differentiable (the current implementation uses scikit-image, which isn't differentiable; a PyTorch SSIM implementation would be needed).

---

### Category 9: Ensemble Methods

**Q26. How could ensemble methods improve this project's results?**

> **Answer**: Several ensemble approaches could help:
> 1. **Model averaging**: Train 3-5 DS-CNN models with different seeds, average their predictions. This reduces variance and could close the accuracy gap with the baseline.
> 2. **Snapshot ensembles**: Use cosine annealing to collect multiple models from different points in the LR cycle (the project already uses cosine annealing!). Average predictions from epochs 50, 60, 70.
> 3. **Baseline + DS-CNN ensemble**: Average predictions from both architectures — the baseline captures fine details, DS-CNN provides efficiency. At inference, run both and average.
> 4. **Boosting-inspired**: Train a second model on the residuals (errors) of the first model.
> 
> The project doesn't use ensembles, which is a missed opportunity for improving accuracy without architectural changes.

**Q27. The project saves checkpoints every 10 epochs. How could these be used for a "poor man's ensemble"?**

> **Answer**: The checkpoints at epochs 10, 20, 30, ..., 70 plus the best model represent different points in the optimization landscape. A **checkpoint ensemble** (or "SWA" — Stochastic Weight Averaging) would: (1) load the last K checkpoints (e.g., epochs 50, 60, 70); (2) average their weights; (3) use the averaged model for inference. This is essentially free — no extra training needed. SWA typically finds flatter minima that generalize better. The project's [`save_checkpoint()`](../src/utils/io.py) already saves all necessary state.

---

### Category 10: Support Vector Machines & Radial Basis Functions

**Q28. How does the U-Net's receptive field relate to the concept of radial basis functions?**

> **Answer**: Each neuron in the U-Net has a **receptive field** — the region of the input image that influences its output. At the bottleneck (after 4 MaxPool2d operations), each neuron's receptive field covers a large portion of the 256×256 input. This is conceptually similar to RBFs: each bottleneck neuron acts like a basis function centered on a spatial region, with the encoder learning the "kernel width" (receptive field size) and the decoder learning how to combine these basis functions to reconstruct the output. The key difference: RBFs use fixed, predefined kernels (e.g., Gaussian), while the U-Net **learns** its spatial kernels end-to-end.

**Q29. Could an SVM solve this problem? Why or why not?**

> **Answer**: An SVM could not practically solve this problem because: (1) **output dimensionality** — SVMs produce scalar outputs, but this task requires 256×256 = 65,536 output values per image; you'd need 65,536 separate SVMs; (2) **input dimensionality** — each input is 65,536 pixels; the kernel matrix would be N×N = 13,178² ≈ 174 billion entries; (3) **spatial structure** — SVMs don't exploit spatial locality the way convolutions do; (4) **scalability** — SVM training is O(N²) to O(N³), infeasible for 13K samples with 65K features. CNNs are fundamentally better suited because they exploit **translation equivariance** and **local connectivity**.

---

### Category 11: Project-Specific Deep Dives

**Q30. Explain the EPE metric. Why is it important for lithography, and how is it computed?**

> **Answer**: Edge Placement Error (EPE) from [`epe.py`](../src/metrics/epe.py) measures how far predicted mask edges are from ground truth edges in pixels. Computation: (1) binarize both masks at threshold 0.5; (2) extract edges via morphological dilation minus erosion; (3) compute distance transform from GT edges; (4) EPE = mean distance of predicted edge pixels to nearest GT edge. It's critical because in semiconductor manufacturing, **edge placement determines circuit functionality** — a few nanometers of error can cause short circuits or open circuits. The baseline achieves EPE=0.0000 at epoch 70 (perfect edge alignment at this resolution), while DS-CNN has EPE=0.0011 pixels.

**Q31. The baseline achieves EPE=0.0000 at epoch 70 but EPE=0.0025 at epoch 100. What explains this?**

> **Answer**: EPE=0.0000 at epoch 70 is likely an artifact of: (1) **resolution** — at 256×256 (downsampled from 2048×2048), sub-pixel edge differences are lost; (2) **binarization threshold** — the 0.5 threshold may be too coarse, causing both predicted and GT edges to snap to the same pixel boundaries. At epoch 100, the model has drifted slightly from its optimal point (the cosine LR is near zero, allowing small random walks), causing edges to shift by ~0.0025 pixels on average. This confirms that **epoch 70 is the optimal checkpoint** and that EPE is sensitive to small prediction changes near the binarization threshold.

**Q32. Why does the project avoid photometric data augmentation?**

> **Answer**: In lithography, pixel intensity represents **optical transmission** — a value of 0.7 means 70% light transmission. Randomly changing brightness would create physically invalid training examples. Only geometric transforms (flip, rotate) are valid because lithography masks have mirror and rotational symmetry. This is a **domain-informed design decision** that demonstrates understanding of the application.

**Q33. How does the data pipeline handle the 2048×2048 → 256×256 resize? What information is lost?**

> **Answer**: Resizing happens during preprocessing (BILINEAR interpolation) and as a safety net at load time. The 8× downsampling loses: (1) **fine edge details** — sub-8-pixel features are aliased; (2) **small features** — features smaller than ~8 pixels in the original are lost; (3) **edge precision** — EPE at 256×256 has 8× coarser resolution than at 2048×2048. This is a significant limitation — the project trades spatial precision for computational feasibility on a 12GB GPU.

**Q34. The LR sweep found LR=1e-3 optimal for DS-CNN. The main training uses LR=1e-4. Why the discrepancy?**

> **Answer**: The LR sweep (10 epochs) found LR=1e-3 converges fastest and achieves the best short-term results. But the main training (70-100 epochs) uses LR=1e-4 with cosine annealing. This is because: (1) **high LR is good for fast convergence but can overshoot** in later epochs; (2) **cosine annealing from 1e-4 to 1e-6** provides a smooth decay that helps find flatter minima; (3) the sweep was only 10 epochs — LR=1e-3 might diverge or oscillate over 70+ epochs. The optimal strategy would be **warmup to 1e-3, then cosine decay** — combining the sweep's finding with the main training's stability.

**Q35. Compare the three DS-CNN width variants. What does this tell us about the accuracy-efficiency Pareto frontier?**

> **Answer**: From the width sweep:
> - **1.0× [64,128,256,512]**: 6M params, 28.5B FLOPs, SSIM=0.966
> - **1.5× [96,192,384,768]**: 13.4M params, 63.3B FLOPs, SSIM=0.971
> - **2.0× [128,256,512,1024]**: 23.8M params, 112.9B FLOPs, SSIM=0.972
> 
> The 1.5× variant is on the **Pareto frontier** — it achieves 99% of the 2.0× accuracy with 56% of the parameters. The 2.0× variant shows **diminishing returns** — 78% more parameters for only 0.1% SSIM improvement. This demonstrates the classic accuracy-efficiency tradeoff curve: initial parameter increases yield large accuracy gains, but returns diminish rapidly.

**Q36. The 100-epoch results show worse metrics than 70-epoch results. What does this tell us about training dynamics?**

> **Answer**: This reveals several important training dynamics: (1) **the cosine LR schedule reaches near-zero LR** at epoch 100 (eta_min=1e-6), meaning the model barely updates but can still drift; (2) **the best checkpoint mechanism is critical** — `best_model.pt` should capture the epoch-70 optimum, but if the 100-epoch run's `best_model.pt` was evaluated, it may reflect a different local minimum; (3) **overfitting accumulates** — with near-zero LR, the model slowly memorizes training noise without improving generalization; (4) **practical implication**: always use the `best_model.pt` checkpoint, not the final epoch checkpoint, and consider stopping training when validation loss plateaus for 10+ epochs.

---

## Part 3: Learning Guide — Connecting Project to Course Concepts

### Study Priority Matrix

| Priority | Topic | Why It Matters for This Project | Key Numbers to Know |
|----------|-------|-------------------------------|---------------------|
| 🔴 High | Neural Networks / CNNs | Core architecture — U-Net, depthwise separable convs | 31M vs 6M params, 5× speedup |
| 🔴 High | Bias-Variance | Explains accuracy gap between models | SSIM: 0.982 vs 0.976 (0.6% gap at epoch 70) |
| 🔴 High | Generalization | StdMetal test, VC theory, overfitting at epoch 100 | SSIM drops 0.971→0.893 on StdMetal |
| 🟡 Medium | Regularization | BatchNorm, augmentation, gradient clipping | No dropout, no weight decay |
| 🟡 Medium | Kernel Methods | Depthwise conv = separable kernel factorization | FLOPs: 109B→28.5B (74% reduction) |
| 🟡 Medium | Loss/Optimization | MSE loss, Adam, cosine annealing, mixed precision | LR=1e-4, cosine to 1e-6, T_max=100 |
| 🟢 Lower | Ensemble Methods | Not implemented but could improve results | Checkpoint ensemble is free |
| 🟢 Lower | SVM/RBF | Contrast with CNN approach | SVM infeasible: 65K features, 13K samples |
| 🟢 Lower | Linear Models | Baseline comparison (why nonlinearity needed) | Optical proximity is nonlinear |

### Quick-Reference Cheat Sheet

```
PROJECT AT A GLANCE
━━━━━━━━━━━━━━━━━━
Task:       Layout → Mask prediction (image-to-image regression)
Dataset:    LithoBench MetalSet, 16,472 tiles, 256×256 grayscale
Split:      80/10/10 (seed=42)
Loss:       MSE
Optimizer:  Adam (lr=1e-4, no weight decay)
Scheduler:  Cosine annealing (T_max=100, min_lr=1e-6)
Epochs:     Trained to 100; best checkpoint at ~epoch 66-70
Batch:      4 × 4 accumulation = effective 16
Precision:  Mixed (float16 forward, float32 params)
GPU:        12GB (SJSU HPC)

BASELINE U-NET (best @ epoch 70)     DS-CNN U-NET (best @ epoch 70)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
31.0M params                         6.0M params (81% fewer)
109.3B FLOPs                         28.5B FLOPs (74% fewer)
19.04 ms inference                   3.78 ms inference (5× faster)
MSE: 0.000046                        MSE: 0.000068 (+48%)
SSIM: 0.9823                         SSIM: 0.9761 (−0.6%)
EPE: 0.0000 px                       EPE: 0.0011 px

AT EPOCH 100 (both models regress slightly):
Baseline MSE: 0.000058, SSIM: 0.9793
DS-CNN MSE: 0.000096, SSIM: 0.9673

KEY INSIGHT: 0.6% SSIM loss buys 81% parameter reduction
             70 epochs is the sweet spot; 100 shows diminishing returns
```

### Common Pitfall Questions (Things Examiners Love to Ask)

1. **"Your EPE is 0.0000 at epoch 70 — does that mean perfect edge placement?"** → No, it's a resolution artifact (256×256 is too coarse). At epoch 100, EPE=0.0025, showing the metric is sensitive to small changes.
2. **"Why not just use a bigger DS-CNN to match baseline accuracy?"** → The 2× wider variant (23.8M params) nearly matches baseline (31M) but is slower (30ms vs 19ms) — defeats the purpose.
3. **"Is 0.6% SSIM difference statistically significant?"** → We can't say — no confidence intervals, single split, single run.
4. **"Why Adam and not SGD?"** → Faster convergence on limited HPC time; SGD might generalize better.
5. **"The 100-epoch results are worse than 70-epoch. Is that a bug?"** → No, it's expected — cosine annealing reaches near-zero LR, and the model drifts. The `best_model.pt` mechanism captures the optimal checkpoint.
6. **"What would you do differently?"** → Add SSIM loss, use weight decay, run k-fold CV, try MobileNetV2 inverted residuals, evaluate at 2048×2048, stop training at epoch 70.
