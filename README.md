# Movie Rating and Popularity Prediction Framework

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a machine learning framework to predict movie **popularity** and **audience rating** using metadata from The Movie Database (TMDB). It compares traditional linear baselines (Ridge, LASSO) with a **Neural Generated Coefficients and Biases (NGCB)** framework, which uses a neural network to dynamically generate regression parameters based on movie context.

## Overview

Predicting movie success is a challenging task due to the complex interplay of various factors. This project investigates whether a movie's **runtime**, conditioned on its **semantic context** (Genre, Production Company, Origin Country, Language), can effectively predict its success metrics.

The project investigates two primary modeling approaches:

1.  **Linear Baseline Models**: Ridge and LASSO regression using multi-hot encoded categorical features.
2.  **Neural Generated Coefficients and Biases (NGCB)**: A "Conditional Linear Regression" approach where a neural network generates the slope ($w$) and bias ($b$) for the runtime feature based on embedding representations of categorical features.

## Dataset

The dataset is a subset of the **TMDB Movie Metadata**, containing approximately **10,000 samples** sampled from a pool of 900,000+ records.

**Input Features**
* `runtime` (Numerical): Duration of the movie.
* `production_company_name` (Categorical, Multi-label)
* `origin_country` (Categorical)
* `original_language` (Categorical)
* `genre_names` (Categorical, Multi-label)

**Target Variables**
* `popularity` (Numerical)
* `vote_average` (Numerical)

## Methodology

### 1. Baseline: Linear System Methods

Standard linear regression with regularization is employed as the baseline.

$$
\hat{Y} = XW + b
$$

* **Ridge Regression**: Utilizes L2 regularization to prevent overfitting.
* **LASSO Regression**: Utilizes L1 regularization for feature selection and sparsity.
* **Feature Engineering**: Multi-hot encoding is applied to all categorical variables.

### 2. Proposed: Neural Generated Coefficients and Biases (NGCB)

In this framework, the neural network functions as a **Coefficient Generator** rather than a direct predictor.

* **Context Representation**: Categorical features are processed through Embedding Layers and aggregated via Mean Pooling to form a dense context vector $\mathbf{c}$.
* **Coefficient Generation**: A Multi-Layer Perceptron (MLP), denoted as $\phi$, takes $\mathbf{c}$ as input and outputs the specific regression parameters for the current instance:
    $$
    (w^p, w^r, b^p, b^r) = \phi(\mathbf{c})
    $$
* **Conditional Prediction**: The final prediction is computed as a linear function of `runtime` ($z$), parameterized by the generated weights:
    $$
    \hat{y} = w \cdot z + b
    $$
* **Loss Function**: The model is trained using a multi-task Mean Squared Error (MSE) loss combined with an L2 penalty on the generated weights.

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* Scikit-learn
* Pandas, NumPy
* Matplotlib

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/wonderingpanda510gh/CPE486_586_Final_Project_Code.git](https://github.com/wonderingpanda510gh/CPE486_586_Final_Project_Code.git)
pip install -r requirements.txt


Usage

To train both the Baseline Linear Models and the Neural NGCB Models, execute the main script:

python main.py
```
Author

Zhehao Yi

University of Alabama in Huntsville

Email: zy0016@uah.edu
