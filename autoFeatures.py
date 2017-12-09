__author__ = 'christiaan'

# coding: utf-8
import pandas as pd
import tensorflow as tf
from sklearn import metrics

#CATEGORICAL_COLUMNS = ["Name", "Sex", "Embarked", "Cabin"]
#CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare", "PassengerId", "Pclass"]

#SURVIVED_COLUMN = "Survived"


def cleanData(train,test):
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    # Impute the missing ages with median age
    train["Age"] = train["Age"].fillna(train["Age"].median()).astype(int)
    test["Age"] = test["Age"].fillna(test["Age"].median()).astype(int)

    # Fill in missing embarked with S
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Embarked"] = test["Embarked"].fillna("S")

    # Fill in missing Cabin with None
    train["Cabin"] = train["Cabin"].fillna("None")
    test["Cabin"] = test["Cabin"].fillna("None")

    # Write our changed dataframes to csv.
    test.to_csv("./test.csv", index=False)
    train.to_csv('./train.csv', index=False)
    return test,train

'''
test,trainX = cleanData('/Users/christiaan/Desktop/train.csv','/Users/christiaan/Desktop/test.csv')

trainX = trainX.drop('Name',axis=1)
trainX = trainX.drop('Ticket',axis=1)
train2 = trainX.drop('Cabin',axis=1)


train = train2.head(300)


test = train2.tail(30)
testLabels = test['Survived']
test.drop("Survived",axis=1,inplace=True)
'''

'''
test = test.drop('Name',axis=1)
test = test.drop('Ticket',axis=1)
test = test.drop('Cabin',axis=1)
'''

def getCategoricalVSContiniousColumns(df):
    analysisTuples=[]
    for var in df.columns:
        analysisTuples.append((var, 1.*df[var].nunique()/df[var].count() < 0.05)) #and (1.*test[var].value_counts(normalize=True).head(top_n) > 0)
    likely_cat = [x[0] for x in analysisTuples if x[1]==True]
    likely_con =[x[0] for x in analysisTuples if x[1]==False]
    return likely_cat,likely_con

def inferCategoryNames(dfColumn):
    #list(map(str,dfColumn.unique()))
    return dfColumn.unique()


def transformToFeatureColumns(dataFrame,train=False,targetColumnName=None):
    if train==True:
      dataFrame = dataFrame.drop(targetColumnName,axis=1)
    cats,cons = getCategoricalVSContiniousColumns(dataFrame)


    feature_cols_cons = [tf.feature_column.numeric_column(varName) for varName in cons]
    feature_cols_cats = [tf.feature_column.categorical_column_with_vocabulary_list(key=varName,vocabulary_list=(inferCategoryNames(dataFrame[varName]))) for varName in cats]

    if train==False:
        return feature_cols_cats,feature_cols_cons
    else:
        return feature_cols_cats,feature_cols_cons



def builtmodel(df,train=False,targetColumnName=None):
    if train==True:
        feature_cols_cats,feature_cols_cons=transformToFeatureColumns(df,train=True,targetColumnName=targetColumnName)


    else:
        feature_cols_cats,feature_cols_cons=transformToFeatureColumns(df)
    wide_columns = [tf.feature_column.embedding_column(feature,dimension=8) for feature in feature_cols_cats]
    deep_columns = [tf.feature_column.embedding_column(feature,dimension=8) for feature in feature_cols_cats]#tf.contrib.layers.embedding_column(x,dimension=8) for x in feature_cols_cats]#feature_cols_cats #TODO

    return tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100,50],
        dnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.01),
        linear_optimizer=tf.train.AdamOptimizer(learning_rate=0.01))



def input_fn(df, labelName,train=False):
  if train:
      labels = df[labelName]
      df = df.drop(labelName,axis=1)
  cats,cons = getCategoricalVSContiniousColumns(df)
  continuous_cols = {k: tf.constant(df[k].values) for k in cons}
  categorical_cols = {k: tf.SparseTensor( indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values, dense_shape=[df[k].size, 1]) for k in cats}
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  if train:
    label = tf.constant(labels.values)
    return feature_cols, label
  else:
    return feature_cols


'''

model = builtmodel(train,train=True,targetColumnName='Survived')

print('training',train)
model.fit(input_fn=lambda: input_fn(train, train=True),steps=2000)
prediction = list(model.predict(input_fn=lambda: input_fn(test)))
labels = testLabels.values
print(prediction)
print(metrics.accuracy_score(prediction,labels))




if __name__ == "__main__":
  tf.app.run()
'''
