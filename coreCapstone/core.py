

from helpers import *
from globalParams import *
from globalImports import *
  
  
# Import data
testFile='defaultCreditCard' # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

if testFile=='defaultCreditCard':
    df=pd.read_csv('C:/Users/plermant/git/sampleFiles/defaultCreditCardClients.csv')


allModels.append(m)
if goFast==0:
    m={'name':'logregress','handle':LogisticRegression(random_state=randomSeed),'type':'sklearn'}
    allModels.append(m)
    m={'name':'ffnn','handle':"defined inside processKeras function",'type':'keras'} # "defined inside processKeras function"
    allModels.append(m)
    m={'name':'iForest','handle':"defined inside processAnomaly function",'type':'anomaly'} # "defined inside processAnomaly function"
    allModels.append(m)
    m={'name':'autoencoder','handle':"defined inside processAnomaly function",'type':'anomaly'} # "defined inside processAnomaly function"
    allModels.append(m)
	
    
if testFile=='defaultCreditCard':
    # set dataset target name 
    target="default"
    numericCat=['SEX','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    # Features to drop as they don't hold any predictive powers
else: 
    print("PICK a file !!!!!!!")
    exit(1)

# print main metrics
print("Main dataframe charactiristics:")
df.info()
df.describe()

# Remove unlabeled samples
before=len(df)
df = df[df[target].notnull()]
print('We removed',before-len(df),'rows that had null target values')

unique, counts = np.unique(df[target], return_counts=True)
print('\nNumber of unique values and percentages for the target',target,':')
val=np.asarray((unique, counts)).T
for i in range(len(val)):
    print(val[i][1],'samples labeled',val[i][0],', or',round(100.0* val[i][1]/len(df[target]) ,2),'percent')
positivePercent=round(1.0* val[1][1]/len(df[target]) ,4)

# drop columns with uniform values
for c in df.columns:
    if len(df[c].unique())==1:
        columnsToDrop.append(c)
		
# removing all features deemed not useful
for c in columnsToDrop: 
    if c in df.columns: # in case this code was already run
        df.drop(columns=c,inplace=True)
        print('Removing feature',c,'because all values are the same or in business parameter list columnsToDrop')
		
# Handle DateTime features
for c in dateTimeList: 
    if debugLevel > 0: print("Turned feature",c,"from object to dateTime")
    df[c]=pd.to_datetime(df[c],errors='raise') # turn object into dateTime

# Handle categories features
for c in df.select_dtypes(include=['object']).columns:
    if c not in dateTimeList and c not in notCategory :
        df[c]=df[c].astype('category')
        if debugLevel > 0: print("Turned feature",c,"from object to category")
		
# Turn discrete numerical values without order meaning into categories
for c in numericCat:
    df[c] = df[c].astype('object')
    if debugLevel > 0: print("Turned feature",c,"from numeric to object")
df.info()

#Let's identify the number of rows with at least one nan/None and the percentage of nulls per columns
nullRows(df)
# Let's check the target values of rows with missing data 
#print('Number of rows with at least one null value with positive target:',len(df[df.isnull().any(axis=1)][df[target]==targetPositive]))
tmp=df[target]
print('Number of rows with null target:',len(tmp[tmp.isnull()]))
colToDrop=nullColumns(df,dropPercent)

# Let's drop the columns with dropPercent or more missing values and see how many rows are still null
#if 'cust_recv_actvtn_chnl_code' in df.columns: df.drop(columns=['cust_recv_actvtn_chnl_code'],inplace=True)
df.drop(columns=colToDrop,inplace=True)
print('\nAfter dropping columns with more than',dropPercent,'% of null values:')
nullRows(df)
nullColumns(df,dropPercent)

# Let's check the target values of rows with missing data 
tmp=df[target]
print('\nNumber of rows with null  target:',len(tmp[tmp.isnull()]))

#out=computeOutliers(df) # available in jupyter, not here
#plotOutliers(out[0],"Low outliers percentage and positive target ratio")
#plotOutliers(out[1],"High outliers percentage and positive target ratio")

#we'll apply Cramer's V to assess correlation between each category feature and the target
catdf=df.select_dtypes(exclude='number')
catDependent=[]
left=[]
err=[]
j=0
for c in catdf.columns:
    j=j+1
    for i in range (len(catdf.columns)):
        if i>=j:
            try:  
                crosstab =np.array(pd.crosstab(catdf[c],catdf[catdf.columns[i]], rownames=None, colnames=None)) # Cross table building
                stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
                obs = np.sum(crosstab) # Number of observations
                mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
                v=(stat/(obs*mini))
                #print('1st column:',c,'2nd column:',catdf.columns[i],'Cramers value:',v)
                if v > cramersThreshold:
                    catDependent.append(catdf.columns[i])
                else:
                    left.append(c)
            except:
                print("X2 could not be computed for feature",c,"with error: The internally computed table of expected frequencies has a zero element")
                err.append(c)
#dedupe
catDependent = list(dict.fromkeys(catDependent))

if len(err) > 0: print("Cramer's V could not be computed for following features,\nwith error: The internally computed table of expected frequencies has a zero element. These features will not be removed:",err)    
if len(catDependent) > 0: print("\nFollowing features will be dropped, as their cramer's V values are over threshold",cramersThreshold,":\n",catDependent)
else: print("No categorical feature will be dropped")

#ordinal encoding
for f in ordinalCategory:
    if f in df.columns: # in case it was already performed
        name=subOrdinal+f
        df[f] = df[f].map(ordinalCategory[f])
        df.rename(columns = {f:name}, inplace = True)
        df[name] = df[name].astype('int64')
        if debugLevel > 0: print("Ordinal encoding feature",f,"into",name)
		
catdf=df.select_dtypes(exclude='number')
numericdf=df.select_dtypes(include='number')
print('numericdf shape',numericdf.shape)
targetdf=df[target]

if target in catdf.columns: catdf=catdf.drop(columns=[target])
print('catdf shape',catdf.shape)

if catdf.shape[1]>0:
    catdfoh=processOneHot(catdf)
    print('catdfoh shape',catdfoh.shape)
	
#  columns with float64 types: can we convert them to int64?
floatColumns=df.select_dtypes(include=['float64']).columns
for c in floatColumns:
    f=False
    v=df[c].values
    for i in range (len(v)):
        if v[i].is_integer() != True:
            f=True # at least one value is not int
            break
    if f==False:
        print('Turning feature',c,'from float64 into int64')
        df[c]=df[c].astype('int64')

if debugLevel > 0: df.info()


#establish baseline thru obvious classifiers
if mainMetric=="roc_auc":
    baseline=.5
elif mainMetric=="recall":
    baseline=.5
elif mainMetric=="f1": # 2*P / (2*P+1) with P percentage of positive labels
    baseline= round(2* positivePercent / (2*positivePercent + 1),4)
print('Baseline is',baseline)

# first round evaluation
if catdf.shape[1]>0:
    df=catdfoh.join(numericdf)
if target not in df.columns:
    df[target]=targetdf
print("df shape is",df.shape)
stageResults['initial']=(processAllModels(df, mainMetric, randomSeed,"",""))
print("\nStage initial results:")
print(stageResults)

#stage power transform
df=powerTransform(df)
stageResults['powerTransform']=(processAllModels(df, mainMetric, randomSeed,"",""))
print("\nStage powerTransform results:")
print(stageResults)

# remove outliers
print(df.shape)
start=df.shape[0]
dfNoOutlier=removeOutliers(df)
outlierTotalCount=start-len(dfNoOutlier)
stageResults['powerTransformNoOL']=(processAllModels(dfNoOutlier, mainMetric, randomSeed,"",""))
print("\nStage powerTransformNoOLresults:")
print(stageResults)

#Pearson for metric columns 
print("Computing Pearson collinearity")
corr = df.corr() # do df.corr('spearman') if switch to spearman
removePearson=[]
j=-1
for c in df.columns:
    j=j+1
    for i in range (len(corr)):
        if (i>j) and (abs(corr[c][i]) >= corrThreshold and (c in numericdf.columns)) : # do not touch non-numeric columns
            removePearson.append(corr.index[i])
            if debugLevel > 0: print("Removing",corr.index[i]," with abs. colinerity with",c,"=",corr[c][i], "greater than threshod",corrThreshold)
 
 #stage colinearity
 # Remove all category columns that are dependent
catdfSlim=catdf.drop(columns=catDependent)
if target in catdfSlim.columns: catdfSlim.drop(columns=[target],inplace=True)
if catdfSlim.shape[1]>0:
    catdfoh=processOneHot(catdfSlim)
    print("catdfoh shape, after removing",len(catDependent),'category features',catdfoh.shape)

# rebuild df without dependent categorical
if catdfSlim.shape[1]>0:
    df=catdfoh.join(numericdf)
if target not in df.columns:
    df[target]=targetdf
print("df shape, after ohe and target are added",df.shape)

#remove numeric collinear
df.drop(columns=removePearson, inplace=True)
print("df shape, after removing",len(removePearson),"numeric collinear features is",df.shape)
colinColumnDropCount=len(catDependent)+len(removePearson)
df=powerTransform(df)
stageResults['collinearity']=(processAllModels(df, mainMetric, randomSeed,"",""))
print("\nStage collinearity results:")
print(stageResults)

# Dropping features flagged by both Permutation and Shap
y=df[target]
X=df.drop(columns=[target])

m=CatBoostClassifier(silent=True,random_state=randomSeed)
m.fit(X,y)

toDropPermutation=featurePermutation(m, X,y, 5, randomSeed, mainMetric,permutationTreshold)
toDropShap=shapExplain(X,y,m,randomSeed)

print("\nFeatures to drop per Permutation:",len(toDropPermutation),"out of total column number =",len(X.columns))
print("Features to drop per Shap:",len(toDropShap),"out of total column number =",len(X.columns))
intersectionDrop= intersection(toDropPermutation,toDropShap)
selectionColumnDropCount=len(intersectionDrop)

# Stage 'tree-based feature selection'
sampler=""
print("Computing metrics for all model for stage featureSelection")
df.drop(columns=intersectionDrop,inplace=True)
print("We dropped",len(intersectionDrop),"intersecting features, out of total",len(X.columns))
stageResults['featureSelection']=processAllModels(df, mainMetric, randomSeed,"","")
print("\nStage featureSelection results:")
print(stageResults)

# resampling stage
# ENN
print("\nStart ENN under-sampling stage")
sampler=EditedNearestNeighbours(sampling_strategy='majority')
stageResults['ENN']=(processAllModels(df, mainMetric, randomSeed,sampler,""))
print("\nStage ENN sub-sampling results:")
print(stageResults)
    
# smote
print("\nStart SMOTE over-sampling stage")
sampler=SMOTE(random_state=randomSeed)
stageResults['SMOTE']=(processAllModels(df, mainMetric, randomSeed,sampler,""))
print("\nStage SMOTE sub-sampling results:")
print(stageResults)

# Based on results above, decide if you should resample or not
resamplerValue= sampler # from above, or
resamplerValue = "" # default, no resampling

# Model itself decides what weight to give to each label value, usually according to imbalance percentage
stageResults['weightLossAuto']=processAllModels(df, mainMetric, randomSeed,resamplerValue,'auto')
print("\nStage weightLossAuto results:")
print(stageResults)
# For comparison, we enforce here a ratio of imbalanceLossWeightRatio, just to compare with default above
stageResults['weightLossCustom']=processAllModels(df, mainMetric, randomSeed,resamplerValue,imbalanceLossWeightRatio)
print("\nStage weightLossCustom results:")
print(stageResults)

# Based on above, pick the appropriate amount of loss weight you want to apply, by default, auto
# note that if resamplerValue value is not empty, then lossWeightValue should be set to empty ("")
lossWeightValue="auto"

# Print final results

stageResultsNormal=copy.deepcopy(stageResults)
for i in stageResults:
    for j in stageResults[i]:
        stageResultsNormal[i][j]=round((stageResults[i][j]-baseline)/(1-baseline),3)
print("Final stageResultsNormal results for test file",testFile)
print("Total number of non-null samples:",df.shape[0],". Number of columns after feature engineering:",df.shape[1])
print("columns dropped by feature collinearity:",colinColumnDropCount,"columns dropped by feature selection:",selectionColumnDropCount,", number of outliers identified:",outlierTotalCount)
print("Baseline value for metric",mainMetric,"is",baseline,". Percent of positive labels is",round(100.0*positivePercent,3),":\n",stageResultsNormal)