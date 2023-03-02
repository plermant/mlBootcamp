from globalParams import *
from globalImports import *

#columnTypeMetrics(df,'object') #Let's take a look at object columns first
def columnTypeMetrics(dfin,ctype):
    if len(dfin.select_dtypes(include=[ctype]).columns) > 0:
        objColumns=dfin.select_dtypes(include=[ctype]).columns
        print("\nMain metrics for the",len(objColumns),ctype,"columns:")
        for c in objColumns:
            print("\nFeature",c, 'is of type',dfin[c].dtypes, 'with following unique values:')
            print(dfin[c].unique())
    else: print('No column of type',ctype)
	

def nullColumns(dfin,dropPercent): #prints non-null percentage of missing values per column
    print('\nColumns with non-zero percent of missing values:')
    input=dfin.isnull().sum()/len(dfin) #holds non-null percentage of missing values per column
    found=0
    res=[]
    if len(input) >0:
        for j in range (len(input)):
            if input[j]>0:
                print(input.index[j],round(100.0*input[j],2),'% of null values')
                if round(100.0*input[j],2) > dropPercent:
                    res.append(input.index[j])
                found=1
    if found==0: print('No column has null values')
    return res
        
def nullRows(dfin): #prints number and percentage of rows with at least one null
    nullRows=dfin[dfin.isnull().any(axis=1)]# rows where there are at least one null
    print('\nNumber of rows with at least one null value is',len(nullRows),', about',round(100.0*len(nullRows)/len(dfin),2),'percent of all rows')
  
# Outliers analysis
def computeOutliers(data):# For each numerical feature, compute the number and percentage of outliers, per inter-quartile rule, and their ratio of positive target
    # We'll use these numbers to decide how to deal with numerical outliers
    resp=[]
    numColumns=data.select_dtypes(include=['float64','int64']).columns
    outLow={} # storing outliers info in the form of {columnName:[lowCount,lowPercent,lowTargetRatio]}
    outHigh={} # storing outliers info in the form of {columnName:[highCount,highPercent,highTargetRatio]}

    for c in numColumns:
        q3 = data[c].quantile(.75)
        q1 = data[c].quantile(.25)
        IQR = q3 - q1
        upper = q3 + outlierIqrRatio * IQR
        lower = q1 - outlierIqrRatio * IQR
        u=data[data[c] > upper]
        l=data[data[c] < lower]
        uCount=len(u)
        uPercent=round(100.0*uCount/len(data[c]),2)
        if len(u) > 0: uTargetRatio=round(100*len(u[u[target]==targetPositive])/len(u),2)
        lCount=len(l)
        lPercent=round(100.0*lCount/len(data[c]),2)
        if len(l) > 0: lTargetRatio=round(100*len(l[l[target]==targetPositive])/len(l),2)
        if uCount>0 and IQR != 0:
    #         print(c,'high value:',upper, ', # of high:',uCount,', % of high:',uPercent)
    #         print("Ratio of positive target for high outliers:",uTargetRatio)
            outHigh[c]=[uCount,uPercent,uTargetRatio]
        if lCount>0 and IQR != 0:
    #         print(c,'Low value:',lower,', # of low:',lCount,', % of low:',lPercent)
    #         print("Percent of positive target for low outliers:",lTargetRatio)
            outLow[c]=[lCount,lPercent,lTargetRatio]    
    #sort from low to high percentages
    outLow=dict(sorted(outLow.items(), key=lambda item: item[1]))
    outHigh=dict(sorted(outHigh.items(), key=lambda item: item[1]))
    resp.append(outLow)
    resp.append(outHigh)
    return resp

def removeOutliers(data): # returns indexes to outlier rows
    remove=data.copy() # we don't want to modify data
    numColumns=remove.select_dtypes(include=['float64','int64']).columns
    for c in numColumns:
        q3 = remove[c].quantile(.75)
        q1 = remove[c].quantile(.25)
        IQR = q3 - q1
        upper = q3 + outlierIqrRatio * IQR
        lower = q1 - outlierIqrRatio * IQR
        if IQR != 0:
            before=remove.shape
            remove = remove[remove[c] < upper]
            remove = remove[remove[c] >  lower]
            if before != remove.shape:
                print('Removing outliers for column=',c)
                print(remove.shape)
    return remove
	
def processOneHot(data):
    col=data.columns
    #first we consolidate all labels beyond dummyMaxCount into label named 'otherLabel'
    for i in range(len(col)):
        data[col[i]] = data[col[i]].astype('object') # need to set type to 'object' else won't be able to set a new label value
        if len(data[col[i]].value_counts()) > dummyMaxCount: # we have too many labels to one-hot encode them all
            tops=data[col[i]].value_counts().index[0:dummyMaxCount]
            print("feature",col[i],"has too many labels, we'll assign the -",otherLabel,"- label to all entries ranked below",dummyMaxCount)
            n=0
            for j in range(len(data)): #we are turning every label past the dummyMaxCount-th into otherLabel
                if(data[col[i]].iloc[j]) not in tops:
                    data[col[i]].iloc[j]=otherLabel      
                    n=n+1
            if debugLevel > 0: print("put",n,"items in", otherLabel," cat, out of total rows=",len(df))

    for i in range(len(col)):
        if debugLevel > 0: print("Feature",col[i],"will be one-hot encoded, resulting in names (one per label):",str(col[i])+"_label, skipping first label")
    
    return(get_dummies(data,drop_first=True,prefix=col,dtype="int64"))
	
def processModelSklearn(model,X_train,y_train,X_test,y_test, metric, seed,w):
    if w=="": model.fit(X_train,y_train)
    else: model.fit(X_train,y_train,sample_weight=w)
    y_pred_proba=model.predict_proba(X_test)
    y_pred=model.predict(X_test)
    if metric=="roc_auc":
        result = round(metrics.roc_auc_score(y_test, y_pred_proba[:, 1]),3)
    elif metric=="recall":
        result = round(metrics.recall_score(y_test, y_pred),3)
    elif metric=="f1":
        result = round(metrics.f1_score(y_test, y_pred),3)
    else: 
        print("We dont support this metric,",metric," aborting")
        exit(1)
    return result
	
def processAnomaly(modelName,X_train,y_train,X_test,y_test, metric, cont):
    if modelName=='iForest':
        clf = IForest(contamination=cont)
    if modelName=='autoencoder':
        clf=AutoEncoder(hidden_neurons=[int(len(X_train.columns)/4), int(len(X_train.columns)/8), int(len(X_train.columns)/8), int(len(X_train.columns)/4)], hidden_activation='relu', optimizer='adam', epochs=epochAutoencoder, batch_size=32, dropout_rate=0.2, validation_size=0.3, preprocessing=False, verbose=debugLevel, random_state=randomSeed, contamination=cont)

    clf.fit(X_train)

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    if (metric=="roc_auc"):
        result=np.round(roc_auc_score(y_test, y_test_scores), decimals=3)
    if (metric=="recall"):
        result=np.round(recall_score(y_test, y_test_pred), decimals=3)
    if (metric=="f1"):
        result=np.round(f1_score(y_test, y_test_pred), decimals=3)
        
    return result
	
# utility functions to define f1 for Keras
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
	
def processKeras(df,model,X,y,X_t,y_t, metric,seed,ratio):
    #hack, for some reason I need to redefine it ... else it uses the old number of features.
    model=tf.keras.Sequential([layers.Dense(int(len(df.columns)/2), activation='relu'),layers.Dense(1,activation='sigmoid')])
    if metric=="roc_auc":
        met=[tf.keras.metrics.AUC()]
    elif metric=="recall":
        met=[tf.keras.metrics.Recall()]
    elif metric=="f1":
        met=[f1_m]
    else: 
        print("We dont support this metric,",metric," aborting")
        exit(1)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=met)
    callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,mode='min')
    if ratio=="":
        model.fit(X, y, epochs=epochCount,verbose=debugLevel,callbacks=[callback],validation_data=(X_t,y_t))
    else:
        weights={0:1.0,1:ratio}
        model.fit(X, y, epochs=epochCount,verbose=debugLevel,class_weight=weights,callbacks=[callback],validation_data=(X_t,y_t))
    print(model.summary())
    result=model.evaluate(x=X_t,y=y_t)
    return round(result[1],3)
	
def processAllModels(data, metric,seed,sampler,w):
    target="default"
    s={}
    #weight can be "", or "auto", or a specific value. we should not have wieghts set alongside sampler param, they are mutually exclusive.
    y=data[target]
    X=data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentHoldout,random_state=randomSeed,stratify=y)
    print("Size of X training set is ",X_train.shape)
    print("Size of X validation set is",X_test.shape)
    print("Target count: negative, positive",y_train.value_counts()[abs(1-targetPositive)],y_train.value_counts()[targetPositive])
    
    if w != "":
        if w == 'auto':
            ratio=round(y_train.value_counts()[0]/y_train.value_counts()[1],2)
        else: ratio= w
        print("Weight mode is",w,"Weight ratio is",ratio)
        weights=np.zeros(len(y_train))
        for i in range(len(y_train)):
            if y_train.iloc[i]==0: weights[i]=1
            else: weights[i]=ratio
                
    if sampler != "":
        if w != "":
            print("cannot set training weights AND resampling, aborting")
            exit(1)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print("After resampling, size of training data is",X_train.shape,". Validation set is unchanged")
        print("Target count: negative, positive",y_train.value_counts()[abs(1-targetPositive)],y_train.value_counts()[targetPositive])
    
    for n in allModels:
        if n['type']=='sklearn':
            if w == "":
                res=processModelSklearn(n['handle'],X_train,y_train,X_test,y_test, metric, seed,"")
            else:
                res=processModelSklearn(n['handle'],X_train,y_train,X_test,y_test, metric, seed,weights)
        elif n['type']=='keras':
            if w == "":
                res=processKeras(data,n['handle'],X_train,y_train,X_test,y_test, metric, seed,"")
            else:
                res=processKeras(data,n['handle'],X_train,y_train,X_test,y_test, metric, seed,ratio)
        elif n['type']=='anomaly':
            contamination =1.0 - len(data[data[target]==0])/len(data) # percentage of outliers
            print("Contamination set to",contamination)
            res = processAnomaly(n['name'],X_train,y_train,X_test,y_test, metric, contamination)
        else: 
            print("WRONG Model Type, exiting ...")
            exit(1)
        print("For classifier",n['name'],metric,"is",res)
        s[n['name']]=res
    
    return s
	
def powerTransform(data):
    if target in data.columns: 
        X = data.drop(columns=[target])
    else: X=data
    xColumns=X.columns
    pt = PowerTransformer(method='yeo-johnson')
    print("Applying powerTransform to all numerical data, make sure to save the lambdas to apply to new incoming data\n")
    # we sometimes get power errors during transform, so we have to handle each column separately and provide fall-back
    for c in xColumns:
        # check that we have entries other than 0 and 1, which we want to leave alone
        u=data[c].unique()
        if (len(u) == 2) and (0 in u and 1 in u): 
            #print("leaving alone because only 0s and 1s:",c)
            continue # leave data alone
        else:
            a=X[c].array.reshape(-1, 1)
            try: tmp=pt.fit_transform(a)
            except: 
                print("Encountered error processing feature",c,", adding 1000 + min to each value and recompute ..")
                add=a.min()+1000
                a=a+add
                tmp=pt.fit_transform(a)
            data[c]=tmp
    return (data)
	
def featurePermutation(cl, X, y, nRepeat, randomSeed, s,mainMetricPermTreshold):
    scoring=[s]
    result = permutation_importance(cl, X, y, n_repeats=nRepeat, random_state=randomSeed, scoring=scoring)
    fig = plt.figure(figsize=(14,40))
    for i in range(len(scoring)):
        plt.subplot(len(scoring),1,i+1)
        plt.subplots_adjust(hspace=1,wspace=0.3)
        forest_importances = pd.Series(result[scoring[i]]['importances_mean'], index=X.columns)
        plt.bar(X.columns, forest_importances, align="center",yerr=result[scoring[i]].importances_std)
        title="Feature importances using permutation for:"+scoring[i]
        plt.title(title)
        plt.xticks(rotation='vertical')
        plt.ylabel("Mean decrease")
        if scoring[i]==mainMetric:
            set=[]
            if debugLevel > 0: print("Features that are below the permutation importance threshold",mainMetricPermTreshold,"for",mainMetric)
            count=0
            total=0
            for j in range(len(forest_importances)):
                total=total+forest_importances[j]
                if forest_importances[j] < mainMetricPermTreshold:
                    print(forest_importances.index[j],forest_importances[j])
                    set.append(forest_importances.index[j])
                    count=count+1
            print("Total feature decrease amounts to",round(total,3))
    #plt.show()
    return set

# return set of columns to drop according to SHAP method and shapThreshold
def shapExplain(X,y,est,s):
    X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
#     explainer = shap.Explainer(est.predict, X100,seed=s)
#     shap_values = explainer(X)
    est.fit(X,y)
    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X)
    sample_ind = 100

#     rf_resultX = pd.DataFrame(shap_values.values, columns = X.columns)
    rf_resultX = pd.DataFrame(shap_values, columns = X.columns)
    vals = np.abs(rf_resultX.values).mean(axis=0)

#    shap_importance = pd.DataFrame(list(zip(X.columns, vals)),columns=['name','importance'])
#    shap_importance.sort_values(by=['importance'],ascending=False, inplace=True)
#    shap.summary_plot(shap_values,features=X)

    set=[]
    for i in range (len(shap_importance)-1):
        if abs(shap_importance['importance'][i]) < shapThreshold:
            set.append(shap_importance['name'][i])
            print("Features below shap importance threshold:",shap_importance['name'][i],shap_importance['importance'][i])
    return set
	
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
	

