__author__ = 'christiaan'
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from autoFeatures import builtmodel
from autoFeatures import input_fn
from tensorflow.contrib import learn
from rnn import lstm_model


trainingSet=pd.read_csv('/Users/christiaan/Desktop/train-file.csv',parse_dates=True,index_col=0)
testSet=pd.read_csv('/Users/christiaan/Desktop/test-file.csv',parse_dates=True,index_col=0)


def extract_time_features_Cat(df,test=False):
    df['year'] = df.index.year- 2015
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayOfWeek']=df.index.weekday
    df['hour']=df.index.hour
    df['dayOfTheYear']= df.index.dayofyear
    if test == False:
        df['IndexTimeSeries']= range(len(df))
    else:
        df['IndexTimeSeries']= range(48120,48120+len(df))
    df['quarterOfYear']=df.index.quarter
    print(df)
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
        df_train['Junction']=df_train['Junction'].apply(int)
        df_train['year']=df_train['year'].apply(int)
        df_train['month']=df_train['month'].apply(int)
        df_train['day']=df_train['day'].apply(int)
        df_train['dayOfWeek']=df_train['dayOfWeek'].apply(int)
        df_train['hour']=df_train['hour'].apply(int)
        df_train['quarterOfYear']=df_train['quarterOfYear'].apply(int)
        df_train['Vehicles_lag24']=df_train['Vehicles_lag24'].apply(int)
        #all_days = pd.date_range(df_train.index.min(), df_train.index.max(), freq='H')
        #df_trainings_y.append(df_train['Vehicles'])
        df_trainings_x.append(df_train)#.drop('Vehicles',axis=1))

    return df_trainings_x


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

df_trainings = pipelineRFR2(trainingSet,testSet,[24])



featureSetSplit = 12300
'''
model = builtmodel(df_trainings[0],train=True,targetColumnName='Vehicles')
labels = df_trainings[0][featureSetSplit:]['Vehicles'].values
model.fit(input_fn=lambda: input_fn(df_trainings[0][:featureSetSplit],'Vehicles', train=True),steps=4500)
prediction = list(model.predict(input_fn=lambda: input_fn(df_trainings[0][featureSetSplit:],'Vehicles')))
print(rmse(prediction,labels))
'''



TIMESTEPS = 5
RNN_LAYERS = [{'steps': TIMESTEPS}, {'steps': TIMESTEPS, 'keep_prob': 0.5}]
DENSE_LAYERS = [2]
TRAINING_STEPS = 200
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100





regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))



regressor.fit(df_trainings[0][:featureSetSplit].drop('Vehicles',axis=1),df_trainings[0][featureSetSplit:]['Vehicles'].values,
                                       steps=TRAINING_STEPS,
                                       batch_size=BATCH_SIZE)


labels = df_trainings[0][featureSetSplit:]['Vehicles'].values
prediction = regressor.predict(df_trainings[0][featureSetSplit:].drop('Vehicles',axis=1))
print(rmse(prediction,labels))