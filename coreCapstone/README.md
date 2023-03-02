Empirical Techniques for Modeling Imbalanced Binary Tabular Datasets

We explore various techniques to predict imbalanced datasets, via feature engineering, data sampling for various machine learning models. 
We do not try to find the very best model (and associated hyperparameters) for a given dataset. 
Rather, we study the impact of different approaches and techniques on the direction of various algoritms. 
We summarized our findings, based on a collection of datasets at https://docs.google.com/document/d/16Hr76mYMeTO8Yvykn6mI376-I3TitgLcNi5_ZkJvhAw

The code can be accessed as a Jupyter notebook or a collection of python files. 
The python version does not offer vizualization, and by default the predictive power is evaluated with roc_auc.
It uses the public dataset https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients.
We would advise you get familiar with the project leveraging the jupyter notebook first so you can tailor it to your dataset.

The requirements.txt file is for a conda virtual environment.

To run, do: "python core.py"
