# Kaggle Playground Series — Season 6, Episode 8

My first Kaggle competition and a hands-on machine learning project focused on building a complete, reproducible ML workflow from raw competition data to Kaggle submission.

## Competition

**Competition:** Playground Series — Season 6, Episode 8
**Platform:** Kaggle
**Sponsor:** Google LLC
**Task:** Predict `addicted_label`
**Competition Type:** Tabular Machine Learning
**Team:** Individual

[Competition Page](https://www.kaggle.com/competitions/playground-series-s6e8)

---

## About This Project

This repository documents my first experience participating in a Kaggle competition.

The main goal is not simply to achieve a high leaderboard score. It is to practice the complete machine learning workflow used in real-world tabular ML problems:

**Understand → Explore → Preprocess → Baseline → Validate → Improve → Submit → Analyze**

I am using this competition to strengthen my practical understanding of data science, machine learning, model evaluation, feature engineering, and experimentation.

---

## Dataset

The competition provides training and test datasets containing behavioral and demographic features related to digital-device and application usage.

The training data contains the target variable:

`addicted_label`

The test dataset does not contain the target and is used to generate Kaggle predictions.

### Main Feature Categories

The dataset includes information related to areas such as:

* Demographics
* Screen time
* Social media usage
* Gaming activity
* Sleep
* Notifications
* Application usage
* Stress
* Academic/work impact
* Other behavioral indicators

The exact feature analysis and preprocessing decisions are documented throughout the notebooks.

> Competition data is not included in this repository. It can be obtained directly from the official Kaggle competition page.

---

## Project Workflow

### 1. Dataset Understanding

* Inspect train and test datasets
* Identify target and ID columns
* Understand feature types
* Check dataset dimensions
* Investigate missing values
* Check duplicates
* Examine target distribution
* Look for suspicious features or potential data leakage

### 2. Exploratory Data Analysis

* Univariate analysis
* Numerical feature distributions
* Categorical feature analysis
* Target relationships
* Missing-value patterns
* Correlation analysis where appropriate
* Identification of unusual observations

### 3. Data Preprocessing

Depending on the findings:

* Missing-value treatment
* Categorical encoding
* Numerical transformations
* Feature selection
* Train/validation preparation

### 4. Baseline Model

A simple baseline model will be established first to create a reference point for future experiments.

The baseline is important because every later improvement should be compared against it.

### 5. Feature Engineering

Potential features will be created based on domain understanding and analysis rather than arbitrary transformations.

Each meaningful feature-engineering experiment will be documented.

### 6. Model Experiments

Different appropriate tabular ML algorithms will be evaluated.

For every major experiment, I will record:

* Model
* Features used
* Preprocessing
* Validation strategy
* Validation score
* Kaggle score
* Observations

### 7. Kaggle Submissions

Predictions will be generated using the test dataset and submitted to Kaggle.

Submission results will be tracked to understand whether improvements in local validation translate into improvements on the leaderboard.

---

## Experiment Tracking

| Experiment   | Model | Validation Score | Kaggle Score | Notes             |
| ------------ | ----- | ---------------: | -----------: | ----------------- |
| Baseline     | TBD   |              TBD |          TBD | Initial reference |
| Experiment 1 | TBD   |              TBD |          TBD | TBD               |
| Experiment 2 | TBD   |              TBD |          TBD | TBD               |

This table will be updated throughout the competition.

---

## Repository Structure

```text
kaggle-playground-s6e8/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── train.py
│
├── submissions/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

The structure may evolve as the project becomes more advanced.

---

## Technologies

* Python
* NumPy
* pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook
* Kaggle

Additional libraries may be added when justified by the experiments.

---

## What I Am Learning

This competition is helping me practice:

* Real-world tabular data analysis
* Exploratory Data Analysis
* Data preprocessing
* Missing-value handling
* Categorical encoding
* Feature engineering
* Train/validation strategies
* Model selection
* Hyperparameter tuning
* Model evaluation
* Kaggle submissions
* Experiment tracking
* Avoiding data leakage
* Reproducible machine learning workflows

---

## Competition Rules

All experiments and submissions are conducted according to the official competition rules.

Important restrictions include:

* Only one Kaggle account may be used.
* Maximum of 10 submissions per day.
* Maximum team size is 3.
* Test/validation records cannot be manually labeled or predicted.
* Competition data must be handled according to the competition's data-use requirements.
* External data and tools must comply with the competition rules.
* Private sharing of competition code outside an official team is not permitted.

See the official competition rules for the complete requirements.

---

## Results

**Final Kaggle Score:** TBD
**Final Rank:** TBD
**Best Submission:** TBD

This section will be updated when the competition is completed.

---

## Final Reflection

This is my first Kaggle competition.

The purpose of this project is to move beyond simply studying machine learning concepts and practice applying them to an actual competitive data science problem.

Rather than focusing only on the final leaderboard position, I am documenting the reasoning behind my experiments, mistakes, improvements, and lessons learned throughout the competition.

**The objective is to become better at solving ML problems, one competition at a time.**

---

## License

This repository contains my own code and analysis.

Competition data is not included. Dataset usage is subject to the competition's terms and license.
