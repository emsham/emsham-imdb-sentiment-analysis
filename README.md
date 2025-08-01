# Sentiment Analysis using TF-IDF and Logistic Regression

A comprehensive implementation of sentiment analysis on movie reviews using TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction and custom Logistic Regression with MAP (Maximum A Posteriori) estimation.

## Overview

This project demonstrates fundamental machine learning concepts through a practical NLP application:

- Custom implementation of logistic regression with L2 regularization
- Stochastic gradient descent with mini-batch optimization
- TF-IDF feature extraction for text representation
- Comprehensive model evaluation and visualization

## Key Features

- **Custom Logistic Regression**: Implementation from scratch with gradient descent optimization
- **MAP Estimation**: Incorporates L2 regularization through Bayesian prior
- **Convergence Monitoring**: Tracks log-likelihood across iterations
- **Hyperparameter Tuning**: Systematic search for optimal regularization parameter (λ)
- **Comprehensive Evaluation**: Accuracy, precision, recall, and F1 metrics
- **Insightful Visualizations**: Parameter influence, convergence plots, and influential word analysis

## Installation

1. Clone this repository:

```bash
git clone [repository-url]
cd sentiment-analysis-tfidf
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `notebook.ipynb` and run all cells to reproduce the analysis

## Implementation Details

### Model Architecture

- **Feature Extraction**: TF-IDF vectorization with English stop words removal
- **Classification**: Logistic regression with sigmoid activation
- **Optimization**: Mini-batch stochastic gradient descent
- **Regularization**: L2 penalty controlled by λ parameter

### Key Functions

- `sigmoid()`: Numerically stable sigmoid implementation
- `beta_map()`: MAP estimation with gradient descent
- `beta_map_sweep()`: Hyperparameter search across λ values
- `log_likelihood()`: Model evaluation metric
- `evaluate_model()`: Comprehensive performance metrics

### Results

- Achieved >85% accuracy on IMDB sentiment classification
- Identified optimal λ through systematic hyperparameter search
- Demonstrated convergence within 200 iterations
- Extracted interpretable word importance features

## Future Improvements

- Implement cross-validation for more robust evaluation
- Add support for multi-class classification
- Explore advanced optimization algorithms (Adam, L-BFGS)
- Implement early stopping based on validation performance
- Add comprehensive unit tests

---

_This project was developed as part of PhD coursework in Probabilistic Models and Machine Learning at Columbia University._
