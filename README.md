# Genetic Algorithms for Searching Ultra-large Libraries

Using genetic algorithms to speed up molecular scoring.

## Local development

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

### Create Environment

The following commands will setup an environment where you can run and test the application locally:

```bash
git clone git@github.com:jonswain/ga-for-ul-libraries.git
cd ga-for-ul-libraries
conda env create -f env.ml
conda activate ga-for-ul-libraries
code .
```

## Procedure

Active learning is used when we have some sort of scoring function that is too computationally expensive to label the full library of compounds. A machine learning model is trained on a subset of the data and used to score all compounds from within the library. The compounds with the best scores from the ML are labelled using the more expensive function, and the labelled data is pooled and used to train a new machine learning model. This cycle is repeated until a finish criteria is met.

## Data

The SMILES data was borrowed from [Thompson Sampling by Pat Walters](https://github.com/PatWalters/TS)