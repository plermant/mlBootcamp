from globalImports import *

# TESTING params, should be all set to zero when custom parameters are set and final run on entire dataset is performed.
goFast=0 #set to 1 for fast processing, only one model used: catboost
debugLevel=0 # typically maps to model verbose levels 0 or 1.

# SYSTEM PARAMETER
# Revisit defaults below according to your dataset and business goals.

# automatically drops columns with more than dropPercent null values
dropPercent=10 
# ratio used to determine outliers - No outlier is dropped by default, regardless of this value setting.
outlierIqrRatio = 1.5
# Cramer's V threshold is uses to assess if 2 categorical features are dependent. If over threshold, they get removed
cramersThreshold=0.9
# we will remove features whose variance multipled by ratio positive/negative label is smaller than varianceThreshold
varianceThreshold = .1
# Parameters to decide how/if to create an 'other' category when performing one-hot encoding on long-tailed distribution
dummyMaxCount = 10 # no more than 10 new features created as a result of one-hot encoding
otherLabel="otherOH" # label name used to consolidates all labels beyond dummyMaxCount
dummyOtherMaxPercent = .05 # the 'other' feature can hold up to this percentage of the distribution.
percentHoldout = 0.3 # percentage of overall dataset to perform validation after training
# main metric to train all models for
mainMetric="roc_auc" # f1 or roc_auc
# correlation threshold, over wich features get removed
corrThreshold = 0.9
permutationTreshold=0.00001 # Will select (and remove) features that are below this threshold thru permutation
shapThreshold=0.001 #  Will select (and remove) features that are below this threshold thru shap
imbalanceLossWeightRatio=10 # in one experiment, we set the loss ratio to give more weight to minority class, with ratio here.
randomSeed=10 # used for reprocibility
tf.keras.utils.set_random_seed(randomSeed)
seed(randomSeed)

epochCount=100 # Large value, however earlyStopping on validation loss is implemented
epochAutoencoder=10 # for autoencoder models only

#model default definition
allModels=[] # holds all the models to be evaluated in this notebook
m={'name':'catboost','handle':CatBoostClassifier(silent=True,random_state=randomSeed),'type':'sklearn'}

# BUSINESS DRIVEN PARAMETERS
# Must be set according to business knowledge
#  Some piror EDA may have to be performed first by setting top doViz to one in top cell.

# Required
target="" 
# Must be 1.  Means that a hit is marked as targetPositive in target column. 
targetPositive=1

# Not required
# Features to drop as they don't hold any predictive powers, like item ID
columnsToDrop=[] # requires business input
# List feature names known to be dateTime 
dateTimeList=[]
# List discrete numeric features that should be categories (nemeric without ranking meaning, like integer color value )
numericCat=[]
#list non-datetime object types known to *not* be categories
notCategory=[] 
# Dictionary of dictionary labels:values , i.e. string categories with rank, like education levels
ordinalCategory={} # e.g. {'education':{'noEdu':0,'highSchool':1,'higher':2},{...}}
# when an ordinal catagory column is replaced by numbers, its name will be appended with subOrdinal featureName becomes featureNamesubOrdinal
subOrdinal="ord_" 

stageResults={} # will hold all the stage resuts, [metric1:{"stageX":{catboost:result,'logRegression':result,"NN":results}},...]