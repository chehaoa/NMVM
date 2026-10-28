# NMVM: A Non-parametric Multi-view Model for Short Text Clustering

[![Paper](https://img.shields.io/badge/KDD-2026-blue)](https://doi.org/10.1145/3770855.3817827)

Official implementation of **[A Non-parametric Multi-view Model for Short Text Clustering](https://doi.org/10.1145/3770855.3817827)**, published at **KDD 2026**, pp. 532–543.

**Enhao Cheng, Xiaolong Zheng, Jintong Li, Juncheng Hou, Fan Liu, Xuemeng Song, Tian Gan, and Jianhua Yin**

NMVM combines sparse bag-of-words features with dense semantic embeddings for short text clustering. It fuses a Dirichlet Process Multinomial Mixture view with a Dirichlet Process Gaussian Mixture view, infers the number of clusters during fitting, and exports representative words for interpreting each cluster.

## Quick start

Use Python **3.10 or newer** (locally verified with Python 3.12). The implementation runs on CPU; a GPU is not required. The bundled Tweet data already includes embeddings, so these commands require **no API key or embedding service**.

Run from the repository root:

```bash
git clone https://github.com/chehaoa/NMVM.git
cd NMVM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m zipfile -e data-GPT3s.zip .

# Short smoke run on the bundled Tweet data
python main.py --dataname Tweet --sampleNum 1 --iterNum 1

# Standard configuration: 15 full Gibbs sweeps, repeated 5 times
python main.py --dataname Tweet --sampleNum 5 --iterNum 15 --seed 2025
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. To inspect all options, run `python main.py --help`. The shell helper runs the bundled Tweet dataset and forwards additional options:

```bash
PYTHON=.venv/bin/python bash run.bash --sampleNum 1 --iterNum 15
```

### Fixed iteration budget

`--iterNum` is the **exact number of full Gibbs sweeps after initialization**. The default is **15**. Fitting never stops based on NMI or any other ground-truth evaluation metric. ACC and NMI are reported after initialization and after the final sweep, solely for evaluation.

This replaces the earlier NMI-based early stopping. A fixed budget may produce different final scores from that earlier implementation; the one-sweep smoke run checks that the pipeline works and is not a paper-results configuration.

## Data

`data-GPT3s.zip` contains **only `data-GPT3s/Tweet`**: 2,472 documents, 89 ground-truth classes, and precomputed 1,536-dimensional embeddings. The repository describes these embeddings as extracted with `text-embedding-3-small`; embedding extraction code is not included.

The paper evaluates six datasets. The remaining five are **not included in this release**:

| Dataset | Domain | Documents | Ground-truth classes | Included |
| --- | --- | ---: | ---: | :---: |
| Tweet | Social media | 2,472 | 89 | Yes |
| News-T | Google News titles | 11,109 | 152 | No |
| News-S | Google News snippets | 11,109 | 152 | No |
| News-TS | Google News titles and snippets | 11,109 | 152 | No |
| StackOverflow | Technical Q&A | 20,000 | 20 | No |
| Biomedical | PubMed titles | 20,000 | 20 | No |

### Using your own data

Provide a UTF-8 JSON Lines file with one document per nonblank line:

```json
{"textCleaned": "example tokenized short text", "embedding": [0.1, 0.2, 0.3], "clusterNo": 1}
```

- `textCleaned`: nonempty, space-separated tokens used by the bag-of-words view. Preprocess text before running NMVM.
- `embedding`: a finite numeric vector with the same dimension for every document, produced by the same embedding model and preprocessing. The short vector above illustrates the format; match the embedding scale and tune hyperparameters for your own data.
- `clusterNo` or `cluster_id`: an integer ground-truth class ID for evaluation. IDs are converted by subtracting one, following the bundled format. Use a consistent label field throughout a file. The current command-line pipeline requires labels, although sampling itself does not use them.

For example, save records to `my-data/custom.jsonl` and run:

```bash
python main.py --dataDir ./my-data --dataname custom --outputPath ./results/custom --sampleNum 1 --iterNum 15
```

The loader first looks for `<dataDir>/<dataname>`, then `<dataDir>/<dataname>.jsonl`. Directory arguments work with or without a trailing slash.

## Configuration

All settings are command-line arguments. Defaults are preserved from the original release:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dataname` | `Tweet` | Dataset filename, with optional `.jsonl` fallback |
| `--dataDir` | `./data-GPT3s/` | Input data directory |
| `--outputPath` | `./results/` | Output directory |
| `--iterNum` | `15` | Fixed full Gibbs sweeps after initialization; positive integer |
| `--sampleNum` | `5` | Repeated runs; positive integer |
| `--seed` | `2025` | Base seed; run `s` uses `seed + s` for the input shuffle, starting at `s = 1` |
| `--K` | `0` | Initial cluster count; the model can create clusters during fitting |
| `--lambda` | `0.6` | Bag-of-words view weight; embedding view weight is `1 - lambda` |
| `--alpha` | `0.1` | Concentration scaling; the implementation uses `alpha × number of documents` |
| `--beta` | `0.05` | Word-distribution prior |
| `--sigma` | `0.0004` | Gaussian-view variance parameter |
| `--kappa` | `0.5` | Gaussian-view prior scaling |
| `--wordsInTopicNum` | `20` | Representative words exported for each nonempty cluster |

The legacy spelling `--_lambda` is also accepted. Use `0 ≤ lambda ≤ 1`, positive `alpha`, `beta`, `sigma`, and `kappa`, and a nonnegative initial `K`. See the paper for the model and hyperparameter definitions. The defaults target the bundled data; they are not universal settings for every embedding model or dataset.

## Results and reproducibility

The program reports the mean and population standard deviation of final per-run ACC and NMI, plus average elapsed seconds per run. Individual runs' initialization and final metrics are written to the log.

Under `<outputPath>/<dataset><parameter-string>/`, each sample writes:

| File suffix | Contents |
| --- | --- |
| `SampleNo<N>ClusteringResult.txt` | Two columns: document ID and predicted cluster ID |
| `SampleNo<N>PhiWordsInTopics.txt` | Representative words and their estimated probabilities for each nonempty cluster |
| `SampleNo<N>SizeOfEachCluster.txt` | Cluster sizes, in cluster-ID order and descending size order |

Document IDs are **1-based positions in the shuffled input**, not original source IDs such as `tweetId`. Cluster IDs are internal IDs and may have gaps. Metrics and inferred cluster counts are also written to `<outputPath>/log.txt`; total elapsed time is appended to `time_NMVM` in the working directory.

Keep the data, Python/package versions, hyperparameters, iteration budget, and seed fixed when comparing runs. Repeating the same command uses the same shuffle seeds. Existing results for the same configuration and sample number are overwritten, so choose a different `--outputPath` to retain separate experiments.

## Development checks

The regression suite uses tiny synthetic CPU inputs and does not require the data archive or an API key:

```bash
python -m unittest discover -s tests -v
```

It checks the fixed iteration budget, independence from ground-truth labels during sampling, command-line arguments, input paths, and output artifacts. GitHub Actions runs these checks on Python 3.10 and 3.12.

## Citation

If you use NMVM in your research, please cite the paper:

```bibtex
@inproceedings{cheng2026nmvm,
  author    = {Cheng, Enhao and Zheng, Xiaolong and Li, Jintong and
               Hou, Juncheng and Liu, Fan and Song, Xuemeng and
               Gan, Tian and Yin, Jianhua},
  title     = {A Non-parametric Multi-view Model for Short Text Clustering},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  pages     = {532--543},
  publisher = {ACM},
  doi       = {10.1145/3770855.3817827},
  url       = {https://doi.org/10.1145/3770855.3817827}
}
```

Machine-readable metadata is provided in [CITATION.cff](CITATION.cff), including the preferred paper citation for GitHub's **Cite this repository** feature.

## Questions and contributions

For reproducibility questions or bug reports, [open an issue](https://github.com/chehaoa/NMVM/issues) with the command, Python/package versions, dataset format, seed, and complete error traceback. Small fixes and improvements are welcome through pull requests.
