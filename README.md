# Building-GPT-from-Scratch

This repository contains a project for the seminar 'Building GPT from Scratch' by Prof. Elia Bruni in the summer term 2025 at the University of Osnabrück. 

## **Table of Contents**
- [Overview](#overview)
- [QuickStart](#quickstart)
- [Structure of this Repository](#structure-of-this-repository)
- [Content of this Repository] (#content-of-this-repository)
---
## Overview


## QuickStart
To start of quickly and be able to execute the code properly, follow this guide.

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
├── main.ipynb
├── README.md
├── requirements.txt
├── models/
└── src/
    ├── 
    └── utils.py

```
- **models/**  
  Contains our trained model files

- **src/**  
  Contains core source code files for agents, environment, training, and utilities.


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

**add picture of output**

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

**Add performance plots here**

The plots show expected trends:
- Increasing k reduces the word-as-token rate, as more words are split into subwords.
- Merge-use rate rises with k, indicating that frequent subwords are effectively merged.
- Normalization affects type compression and multi-character token usage slightly, but overall patterns are consistent.

Overall, these results confirm that our BPE segmenter learns meaningful subword units and generalizes reasonably well to unseen data.

#### Summary 

- Preprocessing included letter-only normalization, word frequency checks, and WebText cleaning.
- BPE was trained for multiple k values and evaluated on multiple datasets.
- Metrics indicate proper segmentation behavior, with decreasing word-as-token rate and increasing multi-character token usage as k grows.

*Full code and additional visualizations are available in [Task 1 Notebook](task1.ipynb) and related Python files.*




