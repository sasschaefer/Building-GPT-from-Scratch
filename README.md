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

### Task 1

#### Words and Corpora: Preprocessing the training data

Before we can actually work with large amounts of texts and training our model, we need to preprocess our text inputs.

And as discussed in class, we started off with replicating the NLTK approach as shown in class.
In this approach, we counted the frequency of the words in a specified portion of shakespeare.txt and sorted them in descending order. 

```
# Complete Linux command for word frequency counting
!cat shakespeare.txt | tr 'A-Z' 'a-z' | tr -sc 'a-z' '\n' | sort | uniq -c | sort -nr | head -10
```

**add picture of output**

Afterwards, we did replicate this approach in python using NLTK. 

#### Byte-Pair Encoding 

Our goal was to implement and train a BPE with a varying k.  The general procedure was as follows:

**describe BPE programming**

We did also test the segmenter against an unseen webtext of the same size as the shakespeare one. In the following plots, we evalueted the performance of our BPE.

** add picture of diagram**

As seen in the plots, ....
Accuracy stuff...

### Task 2







