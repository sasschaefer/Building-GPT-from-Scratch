# Building-GPT-from-Scratch

This repository contains a project for the seminar 'Building GPT from Scratch' by Prof. Elia Bruni in the summer term 2025 at the University of Osnabrück. 

## **Table of Contents**
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Structure of this Repository](#structure-of-this-repository)
---
## Overview


## QuickStart
To start of quickly and be able to execute the code properly, follow this guide. Or you can use this shortcut and have a look at the project in this [Google Colab](
https://colab.research.google.com/drive/1gcrZEx1hgEA_Q6HeN9q6efYLW1Vq-80p?usp=sharing)


Fist we need to install [Git](#git) to be able to clone this repository.
Then decide, whether you want to set up your virtual environment with [venv](#venv) (built into Python) or [Conda](#conda) (a package and environment manager from Anaconda/Miniconda).

### Install Git
<a name="git"></a>
Download and install Git:

- Visit the [official Git website](https://git-scm.com/) to download the latest version of Git.
- Follow the installation instructions for your operating system.

### Clone the Git Repository

- Open a terminal or command prompt.
- Go to the directory where you want to store everything regarding the course:
```bash
cd <directory_name>
```
- Clone the Git repository:
```bash
git clone https://github.com/sasschaefer/Building-GPT-from-Scratch
```
- Change into the cloned repository:
```bash
cd UDL-Reinforcement-Learning
```

### Set Up a Virtual Environment (venv)
<a name="venv"></a>

Download and install Python:
- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:
--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```
- Install required packages
```bash
pip install -r requirements.txt
```

### Set Up a Virtual Environment (conda)
<a name="conda"></a>
- Create a virtual environment:
1. Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux).
2. Navigate to the directory where you saved the environment.yml file. (This should be YOUR_PATH/UDL-Reinforcement-Learning/)
3. Execute the following command to create the environment:

```bash 
conda create -f environment.yml
```
- Activate the virtual environment:
--> On Windows, Unix and MacOS:
```bash
conda activate bgpt
```

## Structure of this Repository

```
.
├── .gitignore
├── task1.ipynb
├── task2.ipynb
├── task3.ipynb
├── task4.ipynb
├── README.md
├── requirements.txt
├── Corpus/
├── Generated_tokens/
├── artifacts/
├── runs/
├── models/
└── src/
    ├── bpe.py
    ├── data_utils.py
    ├── evaluation.py
    ├── neural_ngram.py
    └── ngram.py

```
- **models/**  
  Contains our trained model files

- **src/**  
  Contains core source code files for bpe, evaluation, neural n-grams, ngrams and utilities.


## Content of this Repository

The project was split into four tasks. In this document, we will go through each task separetly and explain our procedure. However, for more detailed implementation information, please have a look at the specified notebook and python files. 

---

### Task 1

Before we can work with large amounts of text and train our model, the raw text inputs need to be preprocessed.  

As discussed in class, we began by replicating a Unix/Linux word frequency command using a portion of `shakespeare.txt`. This allows us to quickly verify the vocabulary and token distributions:
```
# Count word frequencies, lowercase, letters-only, sorted descending
!cat shakespeare.txt | tr 'A-Z' 'a-z' | tr -sc 'a-z' '\n' | sort | uniq -c | sort -nr | head -10
```

We then replicated this approach in Python using NLTK to generate letters-only tokens and compute top-word statistics programmatically. 

#### Byte-Pair Encoding (BPE)

Our main goal was to implement BPE segmentation and train it with varying merge sizes (k). The procedure can be summarized as:

1. **Convert words to symbol sequences:**
    Each word is split into characters with an end-of-word marker </w>.

2. **Count adjacent symbol pairs:**
       ```
        pair_counts = count_adjacent_pairs(word_sequences)
       ```
3. **Merge the most frequent pair:**
       ```
        new_sequences = merge_pair(sequences, most_common_pair)
       ```
4. **Repeat steps 2–3 for k iterations.**
5. **Save vocab and merges for later tokenization and evaluation.**

We also implemented helpers to tokenize new texts using the learned merges and compute metrics such as:

- Average tokens per word
- Word-as-token rate
- Multi-character token usage
- Type compression

#### Evaluation on WebText

To evaluate the generalization of our segmenter, we applied it to an unseen WebText corpus of a similar size. We compared performance on Shakespeare train/test sets and WebText across multiple k values and normalization strategies.


The plots show expected trends:
- Increasing k reduces the word-as-token rate, as more words are split into subwords.
- Merge-use rate rises with k, indicating that frequent subwords are effectively merged.
- Normalization affects type compression and multi-character token usage slightly, but overall patterns are consistent.

Overall, these results confirm that our BPE segmenter learns meaningful subword units and generalizes reasonably well to unseen data.

#### Summary 

- Preprocessing included letter-only normalization, word frequency checks, and WebText cleaning.
- BPE was trained for multiple k values and evaluated on multiple datasets.
- Metrics indicate proper segmentation behavior, with decreasing word-as-token rate and increasing multi-character token usage as k grows.


### Task 2

With BPE-preprocessed text available, we moved on to **training classical n-gram language models** and evaluating their performance under different smoothing techniques.

#### N-gram Modeling  
We implemented models for n = 1…4 with:  
- Maximum Likelihood (ML) estimates  
- Laplace smoothing  
- Linear interpolation (weights tuned on the validation set)  
- Backoff methods: stupid backoff and a simplified Katz-style backoff  

Perplexity (PPL) was the main evaluation metric. For each `k` (BPE merge size), tokenized train/valid/test sets were used, following the class requirement that **validation is for hyperparameter tuning, not k-selection**.

#### Intrinsic Evaluation  
- **Grid search over λ** for interpolation ensured probabilities summed to 1.  
- **Perplexities** were reported across ML, Laplace, interpolation, and backoff.  
- Note: stupid backoff is not normalized, so its PPL values are *relative*, not absolute.


#### Extrinsic Evaluation (Text Generation)  
We extended the models to **generate continuations** from prompts:  
- Decoding via argmax or sampling (temperature scaling).  
- Outputs were scored with **diversity metrics** (`distinct-1/2`) and simple repetition statistics.  
- A fast generation suite compared multiple (prompt, mode, n) configurations systematically.


#### Results  
- **Sampling**: very diverse outputs (distinct-1 ≈ 0.9–1.0, distinct-2 ≈ 0.95–1.0).  
- **Argmax decoding**: collapsed to loops, especially with backoff and Laplace bigrams (d1 ≈ 0.2–0.3). Interpolation with n=3 helped slightly but still repetitive.  
- **Repetition metrics**: adjacent-repeat detector missed phrase-level loops, so richer metrics are recommended.  
- **Backoff caveat**: stupid backoff produced usable generations but should not be compared by PPL to normalized models.

#### Summary  
- Implemented n-gram models with multiple smoothing strategies.  
- Perplexity used for intrinsic evaluation; sampling experiments for extrinsic evaluation.  
- Sampling yielded high diversity, while argmax decoding collapsed.  
- Results highlight the trade-offs between smoothing methods and the limitations of simple backoff.  
- Full details, metrics, and generation outputs are in the Task 2 notebook.

### Task 3
For this task, we implemented a hardcoded neural embedding layer for conditional text generation using only NumPy. The embeddings and training logic were handled manually, with SGD optimization.

We tracked training using perplexity and applied early stopping with patience to prevent overfitting. The best model checkpoints were saved based on validation performance.
The model was able to generate text conditioned on a given context. Loss and validation curves were recorded to visualize training dynamics and assess model performance.

### Task 4
In this task, we extended our previous n-gram models to a small GPT-style transformer for Shakespeare text generation. The focus was on implementing essential components, particularly causal self-attention, manually—without relying on PyTorch’s built-in transformer modules. Full transformer blocks were not reimplemented from scratch, but each block included layer normalization, a manually coded attention mechanism, and an MLP with GELU activation.


We reused BPE merges from the n-gram step and token conventions (<bos>, <eos>, </w>). Training was conducted with standard PyTorch initialization and the AdamW optimizer, keeping optimizer hyperparameters fixed. The model’s embedding size, number of heads, number of layers, batch size, and dropout were adjustable. We implemented logging, CSV exports, checkpointing, and sample text generation during training.


Evaluation metrics included validation and test perplexity, computed using teacher forcing. We also performed a small hyperparameter sweep over dropout rates to study its effect on generalization. Dropout was chosen as a key hyperparameter because it prevents overfitting: too low dropout risks memorization of training data, while too high dropout can hinder learning.


Sample text generation from trained checkpoints demonstrated that the GPT-style model could generate coherent Shakespearean-like sequences conditioned on a prompt, outperforming the earlier n-gram and neural n-gram baselines in terms of perplexity. 



