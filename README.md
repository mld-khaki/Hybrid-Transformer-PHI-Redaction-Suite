# Hybrid Transformer PHI Redaction Suite

A hybrid transformer-based PHI (Protected Health Information) redaction
system combining token-level BIO tagging and character-level supervision
for robust name detection and redaction in TSV clinical documents.

------------------------------------------------------------------------

## Overview

This project provides:

-   **Hybrid Transformer Model**
    -   Token-level BIO classification (B/I/O)
    -   Character-level PHI detection head
    -   Sliding-window inference for long documents
-   **Synthetic Name Augmentation**
    -   Controlled insertion and replacement of names
    -   Per-epoch dataset augmentation
    -   Automatic alignment of redacted counterparts
-   **Batch Redaction Tools**
    -   CLI processing
    -   GUI runner
    -   Smart skipping of unchanged files
    -   Logging support

------------------------------------------------------------------------

## Project Structure

    .
    ├── phi_redactor.py        # Train / eval / predict model
    ├── name_augmentation.py   # Synthetic PHI augmentation
    ├── phi_redactor_gui.py    # GUI batch redaction runner
    ├── phi_redactor_tsv.py    # CLI + GUI TSV processor
    ├── output/                # Saved checkpoints
    └── runs/                  # Training outputs

------------------------------------------------------------------------

## Dataset Format

Expected structure:

    <DATA_ROOT>/
      <CASE_A>/
        org/*.tsv   # Original documents
        red/*.tsv   # Redacted versions (same filenames)
      <CASE_B>/
        org/*.tsv
        red/*.tsv

Redaction masks are automatically derived via alignment between original
and redacted files.

------------------------------------------------------------------------

## Installation

Requires Python 3.9+

Install dependencies:

``` bash
pip install torch transformers numpy tqdm
```

Optional (recommended):

``` bash
pip install accelerate
```

------------------------------------------------------------------------

## Training

``` bash
python phi_redactor.py train \
  --data_root /path/to/data \
  --output_dir ./runs/exp1 \
  --model_name roberta-base \
  --epochs 4 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --char_loss_weight 0.7 \
  --augment
```

### Training with Synthetic Name Augmentation

``` bash
python phi_redactor.py train \
  --data_root /path/to/data \
  --output_dir ./runs/exp1 \
  --names_file additional_names.txt \
  --name_aug_min_pct 0.1 \
  --name_aug_max_pct 3.0
```

------------------------------------------------------------------------

## Evaluation

``` bash
python phi_redactor.py eval \
  --data_root /path/to/data \
  --checkpoint ./runs/exp1/best
```

Metrics include:

-   Character-level precision
-   Recall
-   F1 score
-   Non-PHI retention

------------------------------------------------------------------------

## Prediction (Single File)

``` bash
python phi_redactor.py predict \
  --checkpoint ./runs/exp1/best \
  --input_tsv input.tsv \
  --output_tsv redacted.tsv
```

------------------------------------------------------------------------

## Batch Processing

### GUI

``` bash
python phi_redactor_gui.py
```

### CLI

``` bash
python phi_redactor_tsv.py \
  --input ./input_folder \
  --output ./output_folder \
  --recursive
```

------------------------------------------------------------------------

## Model Architecture

-   Base encoder: RoBERTa / BERT / DeBERTa
-   Token-level BIO classifier
-   Character-level PHI classifier (upsampled from token embeddings)
-   Combined loss:

```{=html}
<!-- -->
```
    Loss = Token_Loss + (char_loss_weight × Char_Loss)

------------------------------------------------------------------------

## Key Features

-   Sliding window inference for long documents
-   Robust to casing and spacing variations
-   Targeted augmentation within name spans
-   Smart file skipping (unchanged outputs not rewritten)
-   Case-based dataset splitting

------------------------------------------------------------------------

## Intended Use

Designed for:

-   Clinical NLP research
-   PHI anonymization pipelines
-   Medical document preprocessing
-   Transformer-based redaction experiments

------------------------------------------------------------------------

## License

Add your preferred license here (MIT recommended for research projects).
