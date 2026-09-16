# Iris Classifier (Decision Tree)

## Overview

End-to-end ML example from Digital Marketing Mastery Module → builds a decision-tree classifier on the classic Iris dataset using scikit-learn.

## Quick start

```bash
git clone https://github.com/alessandroserio66/Iris-Classifier.git
cd Iris-Classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42
```

## Project structure

```text
iris-classifier/
├── data/ # empty – Iris is loaded from scikit-learn
├── notebooks/
│   └── iris_model.ipynb # walk-through notebook
├── src/
│   └── train.py # reproducible CLI script
├── tests/
│   └── test_train.py # basic pytest
├── outputs/ # created automatically (model & figures)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tests

Run the tests with:

```bash
pytest
```