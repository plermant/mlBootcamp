import numpy as np
from numpy.random import seed
import pandas as pd
from pandas import get_dummies
import sweetviz as sw
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,recall_score
import catboost
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
import re
import scipy.stats as stats
from scipy.stats import chi2_contingency , spearmanr
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
import shap
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print
from pyod.models.auto_encoder import AutoEncoder
import copy