import csv
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from numpy import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import OneClassSVM
import json
from sklearn.decomposition import PCA

# load training set
directories = os.listdir('./training')
data_training = {}
label_training = {}
for dir in directories:
    data_training[dir] = pd.DataFrame()
    label_training[dir] = {}
    files = os.listdir(f"./training/{dir}")
    for file in files:
        df = pd.read_csv(f"./training/{dir}/{file}", delimiter=',')
        data_training[dir] = data_training[dir].append(df, ignore_index=True)

    label_training[dir] = data_training[dir].iloc[:, -1].to_numpy()
    data_training[dir] = data_training[dir].iloc[:, :-2].to_numpy()
    label_training[dir][label_training[dir] == 'A'] = 1
    label_training[dir][label_training[dir] == 'N'] = 0
    label_training[dir] = np.array(label_training[dir].tolist())

# load test set
directories = os.listdir('./test')
data_test = {}
label_test = {}
fitness = {}
output_parameters = {}
for dir in directories:
    test_file = os.listdir(f"./test/{dir}")
    data_test[dir] = pd.read_csv(f"./test/{dir}/{test_file[0]}", delimiter=',')

    label_test[dir] = data_test[dir].iloc[:, -1].to_numpy()
    fitness[dir] = data_test[dir].iloc[:, -2].to_numpy()
    data_test[dir] = data_test[dir].iloc[:, :-2].to_numpy()
    label_test[dir][label_test[dir] == 'A'] = 1
    label_test[dir][label_test[dir] == 'N'] = 0
    label_test[dir] = np.array(label_test[dir].tolist())

df = {}
for dir in data_training.keys():
    print(dir)
    df[dir] = {}
    with open("parameters.json", "r") as file_:
        parameters = json.load(file_)
        for alg in parameters[dir].keys():
            if alg == 'fitness':
                y_pred = fitness[dir] < parameters[dir][alg]['threshold']
                df[dir]['fitness'] = ['Fitness',accuracy_score(label_test[dir], y_pred),
                                      precision_score(label_test[dir], y_pred),
                                      recall_score(label_test[dir], y_pred), f1_score(label_test[dir], y_pred)]
            if alg == 'RF':
                RF = RandomForestClassifier(**parameters[dir][alg]['RF'])
                scaler = preprocessing.StandardScaler().fit(data_training[dir])
                data_tr = scaler.transform(data_training[dir])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[dir][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                RF.fit(data_tr, label_training[dir])
                data_ts = scaler.transform(data_test[dir])
                data_ts = pca_opt.transform(data_ts)

                y_pred = RF.predict(data_ts)
                df[dir]['RF'] = ['RF',accuracy_score(label_test[dir], y_pred), precision_score(label_test[dir], y_pred),
                                 recall_score(label_test[dir], y_pred), f1_score(label_test[dir], y_pred)]
            if alg == 'XGboost':
                XGB = xgb.XGBClassifier(**parameters[dir][alg]['XGboost'])
                scaler = preprocessing.StandardScaler().fit(data_training[dir])
                data_tr = scaler.transform(data_training[dir])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[dir][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                XGB.fit(data_tr, label_training[dir])
                data_ts = scaler.transform(data_test[dir])
                data_ts = pca_opt.transform(data_ts)

                y_pred = XGB.predict(data_ts)
                df[dir]['XGboost'] = ['XGboost',accuracy_score(label_test[dir], y_pred), precision_score(label_test[dir], y_pred),
                                      recall_score(label_test[dir], y_pred), f1_score(label_test[dir], y_pred)]
            if alg == 'Adaboost':
                Ada = AdaBoostClassifier(**parameters[dir][alg]['Adaboost'])
                scaler = preprocessing.StandardScaler().fit(data_training[dir])
                data_tr = scaler.transform(data_training[dir])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[dir][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                Ada.fit(data_tr, label_training[dir])
                data_ts = scaler.transform(data_test[dir])
                data_ts = pca_opt.transform(data_ts)

                y_pred = Ada.predict(data_ts)
                df[dir]['Adaboost'] = ['Adaboost',accuracy_score(label_test[dir], y_pred),
                                       precision_score(label_test[dir], y_pred), recall_score(label_test[dir], y_pred),
                                       f1_score(label_test[dir], y_pred)]
            if alg == 'SVM':
                SVM = OneClassSVM(**parameters[dir][alg]['SVM'])
                normal_data = data_training[dir][label_training[dir] == 0]
                scaler = preprocessing.StandardScaler().fit(normal_data)
                data_tr = scaler.transform(normal_data)

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[dir][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                SVM.fit(data_tr)
                data_ts = scaler.transform(data_test[dir])
                data_ts = pca_opt.transform(data_ts)

                y_pred = SVM.predict(data_ts)
                y_true = label_test[dir].copy()
                y_true[y_true == 1] = -1
                y_true[y_true == 0] = 1
                df[dir]['SVM'] = ['SVM',accuracy_score(y_true, y_pred), precision_score(y_true, y_pred),
                                  recall_score(y_true, y_pred), f1_score(y_true, y_pred)]
            if alg == 'IF':
                IF = IsolationForest(**parameters[dir][alg]['IF'])
                scaler = preprocessing.StandardScaler().fit(data_training[dir])
                data_tr = scaler.transform(data_training[dir])

                IF.fit(data_tr)
                data_ts = scaler.transform(data_test[dir])
                y_pred = IF.predict(data_ts)
                y_true = label_test[dir].copy()
                y_true[y_true == 1] = -1
                y_true[y_true == 0] = 1
                df[dir]['IF'] = ['IF',accuracy_score(y_true, y_pred), precision_score(y_true, y_pred),
                                 recall_score(y_true, y_pred), f1_score(y_true, y_pred)]



for pd_alg in df.keys():
    with open('./results/'+str(pd_alg) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for alg in df[pd_alg].keys():
            writer.writerow(df[pd_alg][alg])
