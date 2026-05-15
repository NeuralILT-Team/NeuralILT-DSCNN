# Review Feedback — Final Project Report

**Paper:** *Accelerating Inverse Lithography Technology Using Depthwise Separable CNNs*  
**Reviewer:** Pramod Yadav  
**Date:** May 15, 2026  

---

## Overall Impression

The report provides a well-structured narrative that clearly motivates the problem, explains the DS-CNN approach, and outlines a comprehensive set of experiments. The writing quality is strong and the IEEE conference formatting is appropriate. However, the paper is currently in a **draft state** — it reads well as prose but is missing the quantitative backbone (tables, figures, and precise numbers) that would make it submission-ready. Below is a summary of feedback organized by priority.

---

## 1. Missing Figures and Tables

The paper contains **23 red placeholder markers** where figures and tables should appear. These are critical for a technical paper — reviewers and readers expect to see data, not descriptions of data. All of the underlying numbers already exist in our `RESULTS.md` and experiment logs, so this is primarily a formatting task.

**Suggested priority for insertion:**

| Priority | Item | Source Data |
|----------|------|-------------|
| High | Table: Test performance comparison (MSE, SSIM, EPE) across epochs 20, 70, 100 | `RESULTS.md` — Comparative Analysis section |
| High | Table: Computational efficiency (parameters, FLOPs, runtime) | `RESULTS.md` — Computational Efficiency section |
| High | Figure: Training and validation loss curves | `results/training_curves.png` |
| High | Figure: Prediction grid (input → ground truth → baseline → DS-CNN) | `results/prediction_grids.png` |
| High | Figure: Efficiency bar charts (params + FLOPs) | `results/efficiency_comparison.png` |
| Medium | Table: Learning rate sweep results (5 LRs × 10 epochs) | `RESULTS.md` — LR Sweep section |
| Medium | Table: Width scaling results (1.0×, 1.5×, 2.0×) | `RESULTS.md` — Wider DS-CNN section |
| Medium | Table: Generalization results on StdMetal and StdContact | `RESULTS.md` — Generalization section |
| Medium | Table: Dataset split summary (MetalSet, StdMetal, StdContact) | `README.md` — data description |
| Low | Architecture diagrams (baseline U-Net, DS-CNN, standard vs depthwise conv) | Can be created as TikZ or exported from slides |
| Low | Future architecture diagram | Conceptual — can be deferred |

---

## 2. Numerical Accuracy

Most numbers in the paper are approximately correct, but a few should be tightened up for precision:

- **Parameter reduction:** The paper consistently says "81%." Our actual measurement is **80.6%** (31.0M → 6.0M). Suggest using "approximately 81%" or reporting the exact figure.
- **FLOPs reduction:** The paper says "74%." Actual is **73.9%** (109.3B → 28.5B). Same suggestion.
- **"SSIM within 1% of baseline"** (Discussion section): This is only true at the **epoch 70 checkpoint** (0.64% gap). At epoch 100, the gap widens to **1.23%**, which exceeds the 1% claim. Suggest qualifying this statement with the specific epoch.
- **StdContact sample count:** The paper states 328 samples, but our consistency test evaluated **165 tiles**. We should clarify whether 328 is the full dataset size and 165 is the subset used, or correct the number.

---

## 3. Results Not Yet Reflected in the Paper

Several important findings from our experiments are described in `RESULTS.md` but do not appear in the report:

1. **Best checkpoints occur at non-round epochs** — Baseline peaks at epoch 67 (SSIM=0.982643) and DS-CNN peaks at epoch 66 (SSIM=0.976096). The paper only discusses epochs 20, 70, and 100. Mentioning the actual best checkpoints would strengthen the convergence analysis.

2. **Epoch 100 regression is quantifiable** — Both models show measurable accuracy drops at epoch 100 compared to their peaks:
   - Baseline SSIM: 0.9826 → 0.9793 (−0.34%)
   - DS-CNN SSIM: 0.9761 → 0.9673 (−0.90%)
   - DS-CNN MSE increases by 41% (0.000068 → 0.000096)
   
   This data supports the paper's claim about cosine annealing convergence windows but should be presented quantitatively.

3. **Learning rate sweep data** — The paper describes Experiment 5 qualitatively but the actual numbers are compelling: LR=1e-3 achieves SSIM=0.965 in just 3 epochs, while LR=1e-5 only reaches SSIM=0.335 after 10 epochs. This 19× difference in validation loss deserves a table.

4. **Width scaling trade-off** — The 1.5× DS-CNN (13.4M params) achieves SSIM=0.9714, recovering most of the accuracy gap to the baseline while still using 57% fewer parameters. This is a strong result that should be highlighted.

5. **Consistency test at 100 epochs** — We ran baseline-vs-DS-CNN prediction comparison on StdMetal (SSIM=0.860) and StdContact (SSIM=0.895). This shows the two architectures converge to structurally similar solutions even on unseen data, which supports the paper's generalization claims.

6. **Runtime variability** — DS-CNN runtime varies across checkpoints (16.92ms at epoch 20, 3.78ms at epoch 70, 15.89ms at epoch 100). The "5× speedup" claim is based on the epoch 70 measurement. We should either report the range or clarify which checkpoint is used.

---

## 4. Structural Suggestions

- **Reduce repetition:** The "81% parameter reduction / 74% FLOPs reduction" claim appears in at least 7 places (Abstract, Problem Statement, Solution, Experiments, Results, Discussion, Conclusion). Suggest stating it fully in the Abstract and Results, and referencing those sections elsewhere.

- **Separate methodology from results:** Sections V (Experiments) and VI (Results) currently overlap — both describe what was found. Section V should focus on *how* experiments were set up (datasets, hyperparameters, hardware, evaluation protocol), while Section VI should present *what* was observed (tables, figures, analysis).

- **Reorder Contributions section:** The Contributions section (VII) currently appears between Results and Discussion, which is unusual. Consider moving it after Conclusion or into an Acknowledgments section.

- **Fill in remaining author contributions:** Rana, Pooja, and Rishi subsections are still empty.

- **Fill in remaining author IDs:** Rana and Pooja still show `ID: xxxx`.

---

## 5. Minor Formatting Items

- Section headings use `\textbf{}` wrapping (e.g., `\section{\textbf{Introduction}}`), but `IEEEtran` already bolds section titles automatically. The extra `\textbf{}` can be removed.
- The two equations in the Problem Statement (lines 182–190) lack `\label{}` tags, so they cannot be cross-referenced later in the paper.
- Bibliography entry [b22] (MobileNetV2) uses a different citation format than the rest — it includes "ArXiv.org" as a URL rather than the `arXiv:` prefix style used elsewhere.
- The `\hyperref` package is loaded but `\hypersetup{}` is not configured — consider adding PDF metadata and link coloring.
- `Figures/view.jpg` was added to the repo but is not referenced anywhere in the `.tex` file.

---

## Summary of Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Insert tables with actual experimental numbers from `RESULTS.md` | Team | High |
| 2 | Insert figures (training curves, efficiency charts, prediction grids) | Team | High |
| 3 | Fill in Rana, Pooja, and Rishi contributions + student IDs | Each author | High |
| 4 | Verify and tighten numerical claims (especially SSIM gap, StdContact count) | Team | Medium |
| 5 | Add best-checkpoint analysis (epoch 66/67) and epoch-100 regression data | Team | Medium |
| 6 | Reduce redundancy between Experiments and Results sections | Team | Medium |
| 7 | Add LR sweep and width scaling tables | Team | Medium |
| 8 | Fix minor LaTeX formatting (bold headings, equation labels, bibliography) | Team | Low |

---

*This feedback is intended to help us finalize the report for submission. The core narrative and technical content are solid — the main gap is presenting our actual data in proper tables and figures.*
