# GNN-Final

Final project for the course **Generative Neural Networks for the Sciences** at Heidelberg University.

This project studies **Mechanistic Neural Networks (MNNs)** for scientific machine learning, with a focus on:

- equation identifiability
- robustness under noise and sparsity
- representation learning and generalization
- regularization effects on Lorenz equation discovery

The work is based on the paper and official codebase of **Mechanistic Neural Networks**, and extends the original setup with additional analyses on identifiability and regularization.

---

## Project Overview

Mechanistic Neural Networks (MNNs) aim to bridge neural network flexibility and mechanistic interpretability by learning explicit ODE-based representations rather than purely latent mappings.

In this project, we reproduce key parts of the original MNN framework and then investigate several research questions:

1. **Identifiability and robustness**  
   How reliably can MNNs recover the true governing equations under noisy, sparse, or partially observed conditions?

2. **Representation learning and generalization**  
   How does mechanistic structure affect interpretability, out-of-distribution behavior, and long-horizon forecasting compared to more standard neural architectures?

3. **Lorenz regularization case study**  
   Does sparsity-based regularization, such as \(L_1\) and Elastic Net penalties, improve identifiability in Lorenz equation discovery?

---

## Main Components

This repository contains:

- baseline reproduction of the original MNN implementation
- identifiability experiments on damped sine systems
- architecture and generalization experiments on damped sine and two-body systems
- Lorenz equation discovery experiments
- uncertainty and robustness extensions
- final report and supporting figures

---

## Repository Structure

```text
GNN-Final/
├── figs/                     # figures used in the report
├── results/                  # generated experiment outputs
├── report_gnn.tex            # main LaTeX report
├── references.bib           # bibliography
├── acronyms.tex             # acronym definitions
├── glossary.tex             # glossary
├── hsluthesis.cls           # thesis/report style
├── hsluthesisterms.sty      # additional style file
└── ...
