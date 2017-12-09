__author__ = 'christiaan'


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR
from scipy import stats
from sklearn.svm import LinearSVR


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

trainingSet=pd.read_csv('/Users/christiaan/Desktop/train-file.csv',parse_dates=True,index_col=0)
testSet=pd.read_csv('/Users/christiaan/Desktop/test-file.csv',parse_dates=True,index_col=0)




def plot_series(df):
    df['Vehicles'].plot()
    plt.show()

def extract_time_features_dummies(df):
    df['year'] = df.index.year- df.index[0].year
    df['month'] = df.index.month

    #df[['Jan','Feb','Maa','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]= pd.DataFrame([pd.get_dummies(df.index.month)], index=df.index)

    #df['Jan'],df['Feb'],df['Maa'],df['Apr'],df['May'],df['Jun'],df['Jul'],df['Aug'],df['Sep'],df['Oct'],df['Nov'],df['Dec']= pd.get_dummies(df.index.month)
    dummies = pd.get_dummies(df['month'],columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    #dummies = pd.DataFrame(dummies,axis=1,columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    df2 = pd.concat([df,dummies], axis=1,ignore_index=True)
    print(df2)
    df2.columns = ['Junction', 'Vehicles','ID' , 'year', 'month','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df2.drop(['Dec'],axis=1,inplace=True)
    print(df2)
    df2['day'] = df.index.day
    df2['dayOfWeek']=df.index.weekday
    df2['hour']=df.index.hour
    df['dayOfTheYear']= df.index.dayofyear
    #holidays?
    return df2


def extract_time_features_Cat(df):
    df['year'] = df.index.year- df.index[0].year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayOfWeek']=df.index.weekday
    df['hour']=df.index.hour
    df['dayOfTheYear']= df.index.dayofyear
    #holidays?
    return df

def addLaggingVariables(df,listOfLags):
    for el in listOfLags:
        df['Vehicles_lag'+str(el)] = df['Vehicles'].shift(el)
    df.drop(df.index[:max(listOfLags)], inplace=True)
    return df


def groupByJunction(df):
    listOfJunctions = sorted(df['Junction'].unique())
    return [df.loc[df['Junction']==i] for i in listOfJunctions]


def pipeline(df_train,df_test):
    extendedDf_training = extract_time_features_Cat(df_train)
    dfs_training = groupByJunction(extendedDf_training)

    extendedDf_test = extract_time_features_Cat(df_test)
    dfs_test = groupByJunction(extendedDf_test)

    testperiod = (dfs_test[0].index[0],dfs_test[0].index[-1])
    print(testperiod)

    '''
    for df in dfs:
        plot_series(df)
        FindArimaCoefs(df['Vehicles'])
    '''
    #FindArimaCoefs(dfs[0]['Vehicles'])
    model = ARIMA(dfs_training[0]['Vehicles'].astype(np.float64), order=(7,1,0))
    model_fit = model.fit()
    output = model_fit.predict(testperiod[0],testperiod[1])
    print(output)
    plt.plot(output)
    plt.show()
    print(model_fit.summary())
    return dfs_training


def FindArimaCoefs(df):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df.values.squeeze(),lags=200,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df,lags=200,ax=ax2)

    '''
    for i in range(10):
        for j in range(10):
            try:
                arma_mod42_shift = sm.tsa.ARMA(df,(i,j)).fit()
                print(arma_mod42_shift.summary(),"AIC:",arma_mod42_shift.aic,"BIC: ",arma_mod42_shift.bic)
                print(i,j,"AIC:",arma_mod42_shift.aic,"BIC: ",arma_mod42_shift.bic)
            except:
                print(i,j,'fail')
    '''
    plt.show()


def param_tuning(trainSetX, trainSetY, model, parameters, cvSplits, verbose=False):
    tscv = TimeSeriesSplit(n_splits=cvSplits)
    models = GridSearchCV(model, parameters, cv=tscv, n_jobs=-1)
    models.fit(trainSetX, trainSetY)
    if (verbose):
        print("========== PARAMETER TUNING RESULTS ===========")
        print ("Winner:")
        print (models.best_estimator_)
    return models

def train_model_RFR(trainSetX, trainSetY):
    cvSplits = 5
    maxDepths = [10,15,19]
    max_features=range(3,trainSetX.shape[1])
    nEstimators =[100]
    criterion=['mse']
    n_jobs=[-1]
    parameters = {'max_depth': maxDepths,'n_estimators':nEstimators,'criterion':criterion,'max_features':max_features,'n_jobs':n_jobs}
    RFRs = param_tuning(trainSetX, trainSetY,RFR(),parameters,cvSplits,verbose=True)
    return RFRs.best_estimator_

def train_model_SVM(trainSetX, trainSetY):
    cvSplits = 5
    ParameterTuningRanges=  [{'C': [0.0000001,0.001,0.01,1, 100, 1000],
                              'kernel':['rbf']
                              }]
    svrs = param_tuning(trainSetX, trainSetY, SVR(), ParameterTuningRanges,cvSplits, verbose=True)
    #train_err = -svrs.best_score_
    model = svrs.best_estimator_
    return model

def generate_next_predictionVector(predictionLastWeek,NewFeatureVector):
    augmented_time_series = np.hstack([NewFeatureVector, predictionLastWeek.reshape(len(predictionLastWeek),1)])
    return augmented_time_series

def iterative_forecast(model, x, window, H):
    """ Implements iterative forecasting strategy

    Arguments:
    ----------
        model: scikit-learn model that implements a predict() method
               and is trained on some data x.
        x:     Numpy array containing the time series.
        h:     number of time periods needed for the h-step ahead
               forecast
    """
    forecast = np.zeros(H)
    forecast[0] = model.predict(x)

    for h in range(1, H):
        features = generate_next_predictionVector(x, forecast[:h], window)

        forecast[h] = model.predict(features)

    return forecast

def predictAll(intialLaggedVariables,model,lag,featureVectorsToPredict):
    intialLaggedVariables = intialLaggedVariables
    firstVec = generate_next_predictionVector(intialLaggedVariables,featureVectorsToPredict[0:lag])
    initialpredictions = model.predict(firstVec)

    predictions = [[round(x) for x in initialpredictions]]
    NumberOfTimeSpans= int(len(featureVectorsToPredict)/lag)
    print('numbr of spans',NumberOfTimeSpans)
    for i in range(1,NumberOfTimeSpans):
        if i==1:
            predictionsLastWeek = initialpredictions
        pred = model.predict(generate_next_predictionVector(predictionsLastWeek,featureVectorsToPredict[i*lag:(i+1)*lag]))
        predictions.append([round(x) for x in pred])
        predictionsLastWeek=pred

    if(len(featureVectorsToPredict) % lag !=0):
        pred = model.predict(generate_next_predictionVector(predictionsLastWeek[:(len(featureVectorsToPredict)-NumberOfTimeSpans*lag)],featureVectorsToPredict[NumberOfTimeSpans*lag:len(featureVectorsToPredict)]))
        predictions.append([round(x) for x in pred])


    predictions = [item for sublist in predictions for item in sublist]
    print(predictions)
    return predictions




def pipelineRFR(df_train,df_test,lags):
    extendedDf_training = extract_time_features_Cat(df_train)
    extendedDf_training=addLaggingVariables(extendedDf_training,lags)
    dfs_training = groupByJunction(extendedDf_training)
    df_trainings_x=[]
    df_trainings_y=[]
    for df_train in dfs_training:
        df_train=df_train[(np.abs(stats.zscore(df_train['Vehicles'])) < 3)]
        df_train=df_train.resample('H').bfill()
        #all_days = pd.date_range(df_train.index.min(), df_train.index.max(), freq='H')
        df_trainings_y.append(df_train['Vehicles'])
        df_trainings_x.append(df_train.drop('Vehicles',axis=1))

    bestRFR = train_model_RFR(df_trainings_x[0][:13000], df_trainings_y[0][:13000])

    testSetX=df_trainings_x[0][13025:13200]
    testSetX=testSetX.drop(['Vehicles_lag'+str(lags[0])],axis=1).values
    testSetY=df_trainings_y[0][13025:13200].values
    intialLaggedVariables=testSetY[:lags[0]]
    predictions  = predictAll(intialLaggedVariables,bestRFR,lags[0],testSetX[lags[0]:])
    print('pred',predictions)
    print('testset',testSetY)
    plt.plot(predictions,label='Prediction')
    plt.plot(testSetY,label='Observations')
    print(rmse(predictions, testSetY))
    plt.legend()
    plt.show()


def pipelineRFR2(df_train,df_test,lags):
    extendedDf_training = extract_time_features_Cat(df_train)
    extendedDf_training=addLaggingVariables(extendedDf_training,lags)
    dfs_training = groupByJunction(extendedDf_training)
    df_trainings_x=[]
    df_trainings_y=[]
    for df_train in dfs_training:
        df_train=df_train[(np.abs(stats.zscore(df_train['Vehicles'])) < 3)]
        #df_train=df_train.resample('H').ffill()
        df_train=pd.rolling_mean(df_train.resample("1H", fill_method="ffill"), window=3, min_periods=1)
        #all_days = pd.date_range(df_train.index.min(), df_train.index.max(), freq='H')
        df_trainings_y.append(df_train['Vehicles'])
        df_trainings_x.append(df_train.drop('Vehicles',axis=1))

    for i in range(3,4):#,len(df_trainings_x)):
        splitIndex = int(0.88*len(df_trainings_x[i]))
        print('training models for junction '+str(i))
        bestRFR = train_model_RFR(df_trainings_x[i][:splitIndex], df_trainings_y[i][:splitIndex])

        testSetX=df_trainings_x[i][splitIndex+25:splitIndex+200]
        #testSetX=testSetX.drop(['Vehicles_lag'+str(lags[0])],axis=1).values
        testSetY=df_trainings_y[i][splitIndex+25:splitIndex+200].values


        targets=testSetY[lags[0]:]
        intialLaggedVariables=testSetY[:lags[0]] #take lag-last ones of trainingset
        featureSetToTest=testSetX.drop(['Vehicles_lag'+str(lags[0])],axis=1)[lags[0]:]
        #predictedLabels = featureSetToTest['ID']
        inputFeatures=featureSetToTest.values

        predictions  = predictAll(intialLaggedVariables,bestRFR,lags[0],inputFeatures)
        #print('lenLabels',len(predictedLabels))
        output=list(map(list, zip(*[featureSetToTest['ID'].values,predictions])))
        output = pd.DataFrame(output,columns=['ID','Vehicles'])
        plt.plot(predictions,label='Prediction')
        plt.plot(targets,label='Observations')
        print(rmse(predictions, targets))
        plt.legend()
        plt.show()


def pipelineRFR3(df_train,df_test,lags):
    extendedDf_training = extract_time_features_Cat(df_train)
    extendedDf_test = extract_time_features_Cat(df_test)
    extendedDf_training=addLaggingVariables(extendedDf_training,lags)

    #group training per junction
    dfs_training = groupByJunction(extendedDf_training)
    df_trainings_x=[]
    df_trainings_y=[]
    df_test_x=[]
    for df_train in dfs_training:
        df_train=df_train[(np.abs(stats.zscore(df_train['Vehicles'])) < 3)]
        df_train=pd.rolling_mean(df_train.resample("1H", fill_method="ffill"), window=3, min_periods=1)
        df_trainings_y.append(df_train['Vehicles'])
        df_trainings_x.append(df_train.drop('Vehicles',axis=1))

    #group test per junction:
    dfs_test = groupByJunction(extendedDf_test)
    df_test_x=dfs_test


    result=[]
    for i in range(len(df_trainings_x)):
        print('training model for junction '+str(i))

        gapsize=lags[0]
        trainSetX=df_trainings_x[i][:-gapsize]
        trainSetY=df_trainings_y[i][:-gapsize].values
        bestRFR = train_model_RFR(trainSetX, trainSetY)


        intialLaggedVariables=df_trainings_y[i][-gapsize:].values #take lag-last ones of trainingset

        print('intialLaggedVariables',intialLaggedVariables)


        featureSetToTest=df_test_x[i]
        #predictedLabels = featureSetToTest['ID']
        inputFeatures=featureSetToTest.values
        print('inputFeatures',inputFeatures)

        predictions  = predictAll(intialLaggedVariables,bestRFR,gapsize,inputFeatures)
        output=list(map(list, zip(*[featureSetToTest['ID'].values,predictions])))
        output = pd.DataFrame(output,columns=['ID','Vehicles'])
        result.append(output)
        plt.plot(predictions,label='Prediction')
        plt.legend()
        plt.show()
    resultDF = pd.concat(result)
    print('result',resultDF)
    print('result length',resultDF)
    resultDF.to_csv()

'''
    #train models for each junction
    bestRFR = train_model_RFR(df_trainings_x[0][:10000], df_trainings_y[0][:10000])
    #predict for first week
    #df_0=df_trainings_x[0][1010:1034].drop(['Vehicles_lag'+str(lags[0])],axis=1)
    predictions = bestRFR.predict(df_trainings_x[0][11000:11034])
    targets=df_trainings_y[0][11000:11034]

    #predictAll(intialLaggedVariables,model,lag,featureVectorsToPredict)

    print(rmse(predictions, targets.values))
    print(predictions)
    print(targets)
    #nextweek

'''


''''

#    tscv = TimeSeriesSplit(n_splits=2)
#    test,train = tscv.split(df_trainings_x[0],df_trainings_y[0])
    print(len(df_trainings_x[0]))
    bestRFR = train_model_RFR(df_trainings_x[0][:10000], df_trainings_y[0][:10000])
    print('feature importance:',bestRFR.feature_importances_ )
    predictions = bestRFR.predict(df_trainings_x[0][10010:10030])
    targets=df_trainings_y[0][10010:10030]
    print(predictions)
    print(targets)
    print(rmse(predictions, targets.values))

'''












def generate_features(x, forecast, window):
    """ Concatenates a time series vector x with forecasts from
        the iterated forecasting strategy.

    Arguments:
    ----------
        x:        Numpy array of length T containing the time series.
        forecast: Scalar containing forecast for time T + 1.
        window:   Autoregressive order of the time series model.
    """
    augmented_time_series = np.hstack((x, forecast))

    return augmented_time_series[-window:].reshape(1, -1)



#FindArimaCoefs(trainingSet['Vehicles'])
pipelineRFR3(trainingSet,testSet,[24])
