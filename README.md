# NMVM: A Non-parametric Multi-view Model for Short Text Clustering

This repository contains the official source code for the paper **"A Non-parametric Multi-view Model for Short Text Clustering"**, which has been officially accepted by **KDD 2026**.

NMVM is a novel generative framework specifically designed for short text clustering. It effectively overcomes the sparsity of short texts by seamlessly integrating bag-of-words features with Large Language Model (LLM)-pretrained text embeddings.

## 🌟 Key Features

* **Fully Non-parametric:** Automatically infers the optimal number of clusters directly from the data without requiring a predefined upper bound.
* **Multi-view Fusion:** Integrates the Dirichlet Process Multinomial Mixture (DPMM) model to capture word co-occurrence and the Dirichlet Process Gaussian Mixture Model (DPGMM) to capture dense semantic information.
* **High Efficiency & Scalability:** Employs a novel adaptive stream initialization strategy and an efficient collapsed Gibbs sampling algorithm, leading to rapid convergence and low computational cost.
* **Strong Interpretability:** Automatically identifies and outputs representative words for each cluster, providing measurable evidence of semantic coherence.

---

## 🚀 Quick Start

### Prerequisites

*Please specify your python version and required packages here (e.g., `numpy`, `scikit-learn`, `nltk`).*
*Note: The model utilizes OpenAI's `text-embedding-3-small` for semantic embedding extraction*.

### Running the Model

You can run the NMVM model using the provided entry script:

```bash
python main.py

```

---

## ⚙️ Configuration & Parameters

The behavior of the model can be controlled via various parameters in `main.py`. Below are the primary arguments you can configure:

| Parameter | Description |
| --- | --- |
| `K` | The number of initial clusters to start with. |
| `sampleNum` | The number of repeated experiments to run for robust evaluation. |
| `iterNum` | The number of iteration times for a single test model during Gibbs sampling. |
| `wordsInTopicNum` | The `topN` representative words to output for interpreting each cluster. |

*Note: For advanced tuning, the model also utilizes hyperparameters such as $\lambda$ (balancing the two views), $\alpha$ (pseudo-count for new clusters), $\beta$, and $\kappa$ as detailed in the paper*.

---

## 📊 Datasets

The model has been extensively evaluated on six real-world short text benchmark datasets. You can apply this code to similar data structures:

| Dataset | Domain | Documents | Average Length | Ground-truth Clusters |
| --- | --- | --- | --- | --- |
| **Tweet** | Social Media | 2,472 | 8.56 | 89 |
| **News-T** | Google News (Titles) | 11,109 | 6.23 | 152 |
| **News-S** | Google News (Snippets) | 11,109 | 21.81 | 152 |
| **News-TS** | Google News (Titles + Snippets) | 11,109 | 27.95 | 152 |
| **StackOverflow** | Technical Q&A | 20,000 | 5.42 | 20 |
| **Biomedical** | PubMed Titles | 20,000 | 8.41 | 20 |

---

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing our work:
