````markdown
# Tracegram: Framing Trace-Level Traffic Analysis with Temporally-Aware Multiple Instance Learning

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
  <a href="https://www.usenix.org/conference/usenixsecurity26"><img src="https://img.shields.io/badge/USENIX%20Security-2026-blue"></a>
  <a href="https://github.com/YuchenZhang-Academic/Tracegram"><img src="https://img.shields.io/badge/GitHub-Code-black"></a>
</p>



## Overview

This repository contains the **official implementation of Tracegram**, as presented in:

> **Jian Qu**, **Yuchen Zhang**, **Jialong Zhang**, **Jianfeng Li**, **Xiaobo Ma**  
> **Tracegram: Framing Trace-Level Traffic Analysis with Temporally-Aware Multiple Instance Learning**  
> *Proceedings of the 35th USENIX Security Symposium (USENIX Security 2026)*



**Tracegram** is a **trace-level network traffic analysis framework** that formulates multi-flow behavioral modeling as **Multiple Instance Learning (MIL)** with explicit **temporal awareness**.

Modern network activities—especially in security scenarios such as **intrusion detection, APT analysis, and encrypted traffic monitoring**—are inherently distributed across multiple flows and evolve over extended time intervals. Existing packet-level or flow-level approaches fragment this context and fail to capture cross-flow temporal dependencies.

Tracegram addresses these challenges by:
- Treating a **trace** as a bag and **flows as instances** under an MIL formulation,
- Introducing a **temporally-aware aggregation module** that preserves flow ordering and inter-arrival timing,
- Producing **flow-level attribution signals** that support analyst-oriented investigation and forensics.

The framework is **scalable**, **encoder-agnostic**, and operates effectively on **encrypted traffic** using only header and timing information.



## Framework

Tracegram follows a four-stage hierarchical pipeline:

1. **Packet Tokenization (M0)**  
   Converts raw packets into token sequences while preserving protocol, timing, and payload structure.

2. **Flow Encoder (M1)**  
   Encodes each flow independently using a linear-attention-based backbone to obtain discriminative flow representations.

3. **Temporally-Aware Flow Aggregation (M2)**  
   Aggregates flow representations while modeling order and inter-flow timing, producing a trace fingerprint and flow importance scores.

4. **Trace-Level Classification (M3)**  
   Applies lightweight task-specific heads for trace-level classification, detection, or attribution.

This design avoids the quadratic complexity of monolithic long-sequence Transformers while retaining long-range contextual reasoning across flows.

---

## Repository Structure

```text
Tracegram/
├── dataset_pre_phase0_5/      # Pre-training datasets (Phase 0–0.5)
│   └── dataset/
├── dataset_pre_phase1/        # Pre-training datasets (Phase 1)
│   └── dataset/
├── dataset_test/              # Evaluation datasets
│   ├── label1/
│   └── label2/
├── net3_flow_linear_cls_att/  # Core Tracegram model implementation
├── phase1_cls/                # Phase-1 classification and fine-tuning
│   ├── output/
│   └── pytorch_warmup/
└── README.md
```

## Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* CUDA-enabled GPU recommended (tested on NVIDIA RTX 3080 Ti)
* Common scientific Python packages:

  * numpy
  * scikit-learn
  * tqdm
  * argparse

> Dataset preprocessing assumes that raw traffic traces have been converted into flow-based representations consistent with the paper.


## Datasets

Tracegram is evaluated on four representative datasets covering diverse trace-level analysis tasks:

* **UAV** – User activity classification
* **KWS** – Encrypted keyword search fingerprinting
* **IDI** – IoT device identification
* **ISD** – Intrusion detection

Dataset construction, trace definitions, preprocessing pipelines, and evaluation splits strictly follow the experimental settings described in the paper.

Due to privacy and legal constraints, only processed or anonymized datasets are included in this repository.


## Training and Evaluation

### Flow Encoder Pre-training

The flow encoder is first trained to obtain a **universal flow representation**, following the two-stage supervised tuning strategy described in the paper.

```bash
python train_pretrain.py \
  --data_path dataset_pre_phase1/dataset \
  --output_model output/pretrained_encoder.pt
```

### Trace-Level Fine-tuning

For downstream trace-level classification or detection:

```bash
python train_trace_cls.py \
  --encoder_path output/pretrained_encoder.pt \
  --train_data dataset_test \
  --epochs 50 \
  --batch_size 16
```

The temporally-aware aggregation module automatically produces:

* Trace-level predictions
* Flow-level importance scores for attribution



## Attribution and Interpretability

A key advantage of Tracegram is its **inherent interpretability**.

The aggregation module assigns attention-based importance scores to individual flows, enabling:

* Identification of **key flows** driving a trace-level decision,
* Alignment with known attack phases such as privilege escalation, command-and-control, and data exfiltration,
* Analyst-oriented triage and targeted forensic investigation.

**Note:**
The provided flow-level attribution reflects **correlation rather than causal evidence** and is intended to support investigation and prioritization, not formal proof.



## Code Scope and Disclaimer

This repository is released as a **research reference implementation**.

* The implementation prioritizes **faithfulness to the methodology described in the paper**, rather than production-level optimization.
* Some datasets and preprocessing steps are **environment- and deployment-specific**.
* Reported results in the paper were obtained under controlled experimental settings; exact numerical reproduction may require careful configuration of data splits and parameters.



## Citation

If you use Tracegram in academic work, please cite:

```bibtex
@inproceedings{qu2026tracegram,
  title     = {Tracegram: Framing Trace-Level Traffic Analysis with Temporally-Aware Multiple Instance Learning},
  author    = {Qu, Jian and Zhang, Yuchen and Zhang, Jialong and Li, Jianfeng and Ma, Xiaobo},
  booktitle = {Proceedings of the 35th USENIX Security Symposium},
  year      = {2026}
}
```
