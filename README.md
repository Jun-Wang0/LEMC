# LEMC

**Linguistically enhanced metaphor detection via parallel semantic modulation and contrastive learning**

LEMC is a research implementation for English metaphor detection and downstream sentiment classification. Given a sentence and a target word position, the detector predicts whether the target is used literally (`0`) or metaphorically (`1`). It combines a RoBERTa encoder with explicit lexical features and a supervised contrastive training objective.

The accompanying manuscript is by **Jun Wang and Keng Hoon Gan**, School of Computer Sciences, Universiti Sains Malaysia. The name LEMC denotes **linguistic enhancement (LE)**, **parallel semantic modulation (M)**, and **supervised contrastive learning (C)**.

## Repository contents and code information

| File | Purpose |
| --- | --- |
| [Metaphor_Detection.ipynb](Metaphor_Detection.ipynb) | Loads VUA-20, extracts concreteness and WordNet features, trains the LEMC detector, evaluates predictions, and saves the best checkpoint. |
| [Metaphor_Sentiment.ipynb](Metaphor_Sentiment.ipynb) | First annotates review text with word-level metaphor predictions; then trains a RoBERTa sentiment classifier using those predictions. |
| [README.md](README.md) | Data sources, environment setup, methodology, and instructions for running the notebooks. |

The repository contains notebook source code. Datasets, the concreteness spreadsheet, trained checkpoints, and the manuscript PDF are not bundled. There is no command-line training entry point or `requirements.txt`; configuration is edited directly in the notebook cells.

The released detection notebook defaults to **VUA-20 with `roberta-large`**. The sentiment notebook defaults to **Books review CSV files** and implements concatenation/projection fusion without an attention-based fusion module. The manuscript also reports VUA-18, RoBERTa-base, SST-2, IMDb, baselines, and ablations; separate ready-to-run configurations for all of these experiments are not included.

## Dataset information

### Metaphor detection

The study uses the VU Amsterdam Metaphor Corpus benchmarks, which contain English text from the British National Corpus annotated for metaphor using MIPVU. The manuscript reports these partition sizes:

| Dataset | Partition | Sentences | Annotated tokens | Metaphorical tokens (%) |
| --- | --- | ---: | ---: | ---: |
| VUA-18 | Train | 6,323 | 116,622 | 11.2 |
| VUA-18 | Validation | 1,550 | 38,628 | 11.6 |
| VUA-18 | Test | 2,674 | 50,175 | 12.4 |
| VUA-20 | Train | 10,909 | 160,154 | 12.0 |
| VUA-20 | Test | 3,601 | 22,196 | 17.9 |

The detection notebook downloads the [CreativeLang/vua20_metaphor distribution](https://huggingface.co/datasets/CreativeLang/vua20_metaphor) through Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("CreativeLang/vua20_metaphor", cache_dir="./dataset_cache")
print(dataset)
print(dataset["train"][0])
```

Each example represents one target occurrence. `MetaphorDataset` reads the following fields:

| Field | Required value |
| --- | --- |
| `sentence` | Sentence text; the code obtains words with `sentence.split()`. |
| `w_index` | Zero-based position of the target in that whitespace-split word list. |
| `label` | Integer: `0` for literal, `1` for metaphorical. |
| `POS` | Target word's part-of-speech tag, such as `NOUN` or `VERB`. |

Other fields in the distribution are not required by this loader. Preserve the original token spacing and target indices when preparing another dataset. For VUA-18, obtain the appropriate shared-task data and adapt the loader to this schema; the notebook does not switch datasets automatically. See the [2018 shared-task report](https://aclanthology.org/W18-0907/) and [2020 shared-task report](https://aclanthology.org/2020.figlang-1.3/) for benchmark details.

**Evaluation split:** the current detection code assigns `dataset["test"]` to `val_data` and selects the best checkpoint using its F1. Its printed "Validation Metrics" therefore refer to that test partition. For an independent held-out evaluation, create a development split from the training data, use it for checkpoint selection, and evaluate the test partition only after selection.

### Sentiment classification

| Dataset in the manuscript | Source and evaluation scope |
| --- | --- |
| SST-2 | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2); reported scores use the validation partition, not the official test set. |
| IMDb | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb); 25,000 training and 25,000 test reviews. |
| Amazon Books | [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023); a study-specific sample of 70,000 reviews, split into 56,000 training and 14,000 test examples. |

The sentiment notebook reads local CSV files; it does not download or construct these sentiment datasets. The original Books sample, its selection procedure, and its rating-to-binary-label conversion are not included, so the precise study subset cannot be reconstructed from these notebooks alone.

For the annotation stage, prepare `train.csv` and `test.csv` with:

| Column | Required value |
| --- | --- |
| `text` | Non-empty review or sentence text. |
| `label` | Integer `0` or `1`, with a documented sentiment mapping used consistently across partitions. The example below uses `0` = negative and `1` = positive. |

For example, this is a format illustration, not a record from the study:

```csv
text,label
"The story was absorbing and beautifully written.",1
"The plot was dull and disappointing.",0
```

To use SST-2, export its `sentence` column as `text` and use its validation partition for the manuscript's evaluation setting. For IMDb, use the labeled train/test partitions and retain `text` and `label`. Pass these CSVs through the same annotation stage before sentiment training.

### Lexical resources

- **Concreteness ratings:** obtain the resource associated with [Brysbaert, Warriner, and Kuperman (2014)](https://doi.org/10.3758/s13428-013-0403-5). The notebooks expect an Excel workbook with columns named exactly `Word` and `Conc.M`. If the downloaded resource is in another format, convert it to `.xlsx` while preserving these columns and the original scores. Ratings range from 1 to 5; words missing from the lookup receive `3.0`.
- **WordNet:** downloaded through NLTK. The code uses the first returned synset, its definition, shortest-path distances, and lowest-common-hypernym depths.
- **spaCy English model:** `en_core_web_sm` supplies POS tags during sentiment-data annotation.

## Requirements and installation

Use Python 3.12 with JupyterLab, Jupyter Notebook, or Google Colab as a starting environment. Saved detection-notebook output records Python 3.12 and `datasets==4.4.1`, but does not establish a complete pinned environment. A CUDA-capable GPU is recommended for training `roberta-large`; both notebooks select CUDA when available and otherwise use CPU. Initial execution needs network access to download pretrained models, datasets, and NLP resources.

| Dependency | Use |
| --- | --- |
| `torch` | Models, training, GPU execution, and checkpoints. |
| `transformers` | RoBERTa models, tokenizers, configuration, and learning-rate scheduling. |
| `datasets` | VUA-20 download and loading. |
| `pandas`, `numpy` | CSV/Excel processing and numerical operations. |
| `openpyxl` | Reading the concreteness `.xlsx` workbook with pandas. |
| `scikit-learn` | Accuracy, precision, recall, and F1. |
| `nltk` | WordNet resources. |
| `spacy` | POS tagging for review annotation. |
| `tqdm` | Progress bars. |
| `jupyterlab`, `ipykernel` | Local notebook execution. |

Clone the repository and create an environment:

```bash
git clone https://github.com/Jun-Wang0/LEMC.git
cd LEMC
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Install PyTorch using the command appropriate for your hardware from the [official installation guide](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m pip install "transformers>=4.40,<5" "datasets==4.4.1" pandas numpy openpyxl scikit-learn nltk spacy tqdm jupyterlab ipykernel
python -m nltk.downloader wordnet punkt
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name lemc --display-name "Python (LEMC)"
python -m jupyterlab
```

The Transformers 4.x range is a setup starting point for the notebook API, not a verified original version or a guarantee of exact reproduction. Select the `Python (LEMC)` kernel in Jupyter. In Colab, install packages in the notebook runtime and mount Google Drive if retaining the existing Drive paths. The detection notebook's first cell runs `!pip install -U datasets`; skip or adjust it if you want to keep a pinned environment.

## Usage instructions

Run the stages in this order: **train the detector → annotate sentiment data → train the sentiment classifier**. Edit configuration before executing the large code cells, because each includes an `if __name__ == "__main__": main()` call that starts its pipeline immediately.

### 1. Prepare local paths

The following is a suggested working layout. Only the two notebooks and README are supplied by the repository; create the resource/data directories and supply their contents locally. Model and annotation outputs are generated during execution.

```text
LEMC/
├── Metaphor_Detection.ipynb
├── Metaphor_Sentiment.ipynb
├── README.md
├── resources/
│   └── Concreteness ratings.xlsx
├── data/
│   └── books/
│       ├── train.csv
│       └── test.csv
├── checkpoints/
│   └── metaphor/
└── outputs/
    └── sentiment/
```

Relative paths below assume that the notebook working directory is the repository root. Replace the existing `/content/drive/MyDrive/...` paths as follows, or use your own absolute paths:

| Notebook / location | Setting | Example replacement |
| --- | --- | --- |
| Detection / `main()` | `path_to_concreteness_data` | `"./resources/Concreteness ratings.xlsx"` |
| Detection / `train_model()` | `model_save_dir` | `"./checkpoints/metaphor"` |
| Sentiment / first `main()` | `model_path` | `"./checkpoints/metaphor"` |
| Sentiment / first `main()` | `concreteness_path` | `"./resources/Concreteness ratings.xlsx"` |
| Sentiment / first `main()` | `book_dir` | `"./data/books"` |
| Sentiment / first `main()` | `save_dir` | `"./outputs/sentiment"` |
| Sentiment / second `main()` | `TRAIN_FILE` | `"./outputs/sentiment/train_book_with_metaphor_pos.csv"` |
| Sentiment / second `main()` | `TEST_FILE` | `"./outputs/sentiment/test_book_with_metaphor_pos.csv"` |

These edits connect the stages: the original annotation checkpoint path differs from the detector's save path, and the original sentiment-training filenames differ from the annotation outputs.

### 2. Train and evaluate the metaphor detector

Open `Metaphor_Detection.ipynb` and configure the paths. If you followed the pinned installation above, skip the `!pip install -U datasets` cell; otherwise use it to install/update Datasets. Then execute the detection program cell. The main pipeline:

1. Loads the concreteness workbook and VUA-20.
2. Builds the RoBERTa tokenizer and LEMC model, adding `[TARGET]`, `[/TARGET]`, `[LITERAL]`, and `[POS]` tokens.
3. Constructs target-specific inputs and six lexical features.
4. Uses inverse-class-frequency weighted sampling for training.
5. Trains for four epochs, printing losses and evaluation accuracy, precision, recall, and F1. Precision, recall, and F1 use the metaphorical class (`1`) as positive.
6. Saves the model when the evaluation F1 improves.

The checkpoint directory contains the saved encoder/configuration, tokenizer files, and `pytorch_model.bin`, which is the **full custom `MetaphorRoberta` state dictionary**. Keep these files together. The custom checkpoint must be loaded into the matching model class; loading only a plain `RobertaModel` does not restore the linguistic, POS, modulation, and classifier layers.

To experiment with RoBERTa-base, change the call in `main()` to `build_model(model_name="roberta-base")` and train a matching checkpoint. This selects a different encoder; it does not automatically reproduce every base-model experiment from the manuscript.

### 3. Generate metaphor features for sentiment data

Open `Metaphor_Sentiment.ipynb` and run the first code section, under **Predict and save metaphor_features (Train and Test Sets)**, after updating its paths. It loads the trained detector, predicts one metaphor label per whitespace-delimited review word, obtains POS and lexical features, and writes:

- `train_book_with_metaphor_pos.csv`
- `test_book_with_metaphor_pos.csv`

The output columns are:

| Column | Contents |
| --- | --- |
| `review` | Original input text. |
| `label` | Original binary sentiment label. |
| `metaphor_labels` | String representation of a list of word-level `0`/`1` predictions. |
| `concreteness_features` | String representation of the six lexical features averaged across target words. |
| `pos_tags` | String representation of spaCy-tokenized POS tags, truncated/padded with `PAD` to the configured length; these may not align one-for-one with whitespace-based metaphor labels. |
| `w_indices` | String representation of the original word-index list. |

This stage performs detector inference separately for each word, so it can take substantially longer than a single prediction per review. Reject empty or missing review text before annotation. Inspect a small subset first and verify the word/label counts after parsing the stored list:

```python
import ast
import pandas as pd

annotated = pd.read_csv("./outputs/sentiment/train_book_with_metaphor_pos.csv")
for _, row in annotated.iterrows():
    labels = ast.literal_eval(row["metaphor_labels"])
    assert len(labels) == len(row["review"].split())
    assert all(label in (0, 1) for label in labels)
```

### 4. Train and evaluate the sentiment classifier

Set `TRAIN_FILE` and `TEST_FILE` in the second code section to the generated CSVs, then run the section under **Sentiment Analysis using metaphor_features**.

`MetaphorSentimentDataset` reads `review`, `label`, and `metaphor_labels`; it also accepts `sentence` and `sentiment` as aliases for the first two columns. A fast RoBERTa tokenizer aligns the word-level metaphor labels to subword tokens. `RobertaMetaphorNoAttention` embeds those binary labels, concatenates them with RoBERTa token representations, applies a projection, and uses masked mean pooling for classification. The exported POS tags and concreteness averages are not inputs to this classifier.

The notebook prints accuracy, precision, recall, and F1 after each epoch, treating label `1` as the positive class for precision/recall/F1. The variable named `TEST_FILE` supplies this repeated evaluation set; use a separate development set for model selection when reserving a final test set. The checkpoint-saving line in this section is commented out, so sentiment weights are **not saved by default**. To retain them, enable saving at the best-F1 branch and save the matching tokenizer/configuration as well.

### Default configuration in the released code

| Setting | Metaphor detection | Sentiment classification |
| --- | --- | --- |
| Encoder | `roberta-large` | `roberta-large` |
| Maximum sequence length | 128 | 128 |
| Batch size | 32 | 32 |
| Epochs | 4 | 4 |
| Optimizer | AdamW | AdamW |
| Learning rate | `2e-5` | `1e-5` |
| Warm-up | 10% of training steps | 10% of training steps |
| Gradient clipping | 1.0 | 1.0 |
| Custom dropout | 0.1 | 0.1 |

Detector-specific defaults are POS embedding dimension `64`, semantic projection/controller hidden dimension `128`, linguistic projection `6 → 32 → 16`, contrastive weight `0.1`, and effective contrastive temperature `0.5`. The standalone `supcon_loss` function declares `0.07` as its default temperature, but the model passes `self.temperature=0.5`. Its candidate-selection ratio is `0.5` and focal exponent is `1.5`.

If GPU memory is insufficient, reduce the `DataLoader` batch sizes or use `roberta-base`. These changes alter the experimental setting; reducing the detection batch size also changes the pairs available to the contrastive objective.

## Methodology

1. **Target-specific preprocessing:** identify the target by its word index, surround it with target markers, and append the first WordNet synset definition after `[LITERAL]` when available. Encode this input and the isolated target with a shared RoBERTa encoder. The implementation additionally applies an eight-head self-attention layer to the contextual sequence.
2. **Linguistic enhancement (LE):** compute target concreteness, mean context concreteness, their absolute difference, mean WordNet path distance, mean lowest-common-hypernym depth, and a distance-above-5 indicator. Project these six scalars into a 16-dimensional representation.
3. **Parallel semantic modulation (M):** combine a POS-augmented contextual target representation with the isolated target for the MIP-inspired pathway, and with the sentence representation for the SPV-inspired pathway. Two independent sigmoid gates weight the pathways; their weights need not sum to one. Concatenate the weighted semantic representation with the linguistic representation.
4. **Supervised contrastive learning (C):** optimize negative log-likelihood plus a weighted auxiliary contrastive objective on the fused features. The implementation uses same-label positives, masked top-k candidate selection, focal weighting, and an additional selected-pair term. The selection operates on zero-masked scores, so selected candidates are not necessarily all opposite-label examples.
5. **Downstream application:** use the trained detector's binary word predictions as learned auxiliary embeddings for sentiment classification.

The isolated-target representation and lexical features are approximations to basic meaning and semantic compatibility, rather than verified word senses or direct measurements of human metaphor processing.

## Reported results and reproducibility notes

The following F1 scores (%) are reported in the accompanying manuscript. They are reference values, not results from a fresh run of this README's setup.

| Model | VUA-18 F1 | VUA-20 F1 |
| --- | ---: | ---: |
| RoBERTa-base | 78.0 | 69.5 |
| RoBERTa-base + LEMC | 78.7 | 72.1 |
| RoBERTa-large | 79.5 | 72.8 |
| RoBERTa-large + LEMC | 79.6 | 73.9 |

| Sentiment configuration | SST-2 validation F1 | IMDb test F1 | Sampled Books test F1 |
| --- | ---: | ---: | ---: |
| RoBERTa-large | 95.69 | 92.89 | 94.83 |
| RoBERTa-large + metaphor features | 96.51 | 92.95 | 95.06 |

The manuscript reports point estimates without uncertainty intervals or significance tests. Small score differences should be interpreted descriptively. Exact reproduction also requires the original data preparation, model-selection settings, environment, and checkpoints.

Additional implementation details to account for:

- Neither notebook fixes all random seeds or includes a complete dependency lock file. Record seeds, package versions, dataset revisions, and split membership for new experiments.
- Both tasks truncate inputs to 128 tokens. In detection and annotation, truncation can remove a marked target; inspect target masks before applying the pipeline to long text.
- The two `ConcretenessScorer` implementations differ in their WordNet distance fallback. Detection preserves a valid zero distance, while sentiment annotation uses `distance or 10.0`, which replaces zero as well as missing distances. For consistent features, use the detection notebook's explicit `None` check in the annotation implementation.
- Reuse the checkpoint's tokenizer, vocabulary size, and POS mapping. When loading a GPU-produced state dictionary on CPU, add `map_location="cpu"` to `torch.load`.
- For a new detection dataset, establish any extended POS-to-ID mapping before constructing the model. The notebook currently builds the model before the dataset can add unseen tags, which can produce out-of-range POS embedding indices.
- Notebook output cells are saved execution records and may come from earlier code revisions. The current source and its configuration determine a new run's behavior.

## Citations

If you use LEMC, cite the repository and accompanying manuscript:

> Jun Wang and Keng Hoon Gan. *LEMC: Linguistically enhanced metaphor detection via parallel semantic modulation and contrastive learning*. Research manuscript.

```bibtex
@misc{wang_gan_lemc,
  author = {Wang, Jun and Gan, Keng Hoon},
  title = {{LEMC}: Linguistically enhanced metaphor detection via parallel semantic modulation and contrastive learning},
  howpublished = {Research manuscript and code repository},
  url = {https://github.com/Jun-Wang0/LEMC}
}
```

Publication year, journal details, and DOI should be added when a final bibliographic record is available. Also cite the original resources used in your experiments:

- Leong, Klebanov, and Shutova (2018). [A Report on the 2018 VUA Metaphor Detection Shared Task](https://aclanthology.org/W18-0907/).
- Leong et al. (2020). [A Report on the 2020 VUA and TOEFL Metaphor Detection Shared Task](https://aclanthology.org/2020.figlang-1.3/).
- Brysbaert, Warriner, and Kuperman (2014). [Concreteness ratings for 40 thousand generally known English word lemmas](https://doi.org/10.3758/s13428-013-0403-5).
- WordNet: see the resource's [citation guidance](https://wordnet.princeton.edu/citing-wordnet).
- Socher et al. (2013). [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170/).
- Maas et al. (2011). [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/).
- Hou et al. (2024). [Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](https://arxiv.org/abs/2403.03952).
- Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
- Khosla et al. (2020). [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

## License and contribution guidelines

No repository-level `LICENSE` file has been provided. Contact the maintainers through [GitHub Issues](https://github.com/Jun-Wang0/LEMC/issues) to clarify code reuse and redistribution terms. Third-party datasets, lexical resources, and pretrained models retain their own licenses and conditions; consult their source pages.

For a bug report or proposed improvement, open an issue or submit a pull request. Include the affected notebook/section, environment and package versions, dataset/split, configuration, and a minimal example or error trace. Document any changes to preprocessing or evaluation so that their effects can be assessed, and avoid adding local datasets, model checkpoints, or machine-specific paths to a contribution.
