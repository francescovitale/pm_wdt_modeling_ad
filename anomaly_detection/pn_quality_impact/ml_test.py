import csv
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
import os
import numpy as np

# load training set
directories = os.listdir('./cc_diagnoses')
data_training = {}
label_training = {}
for pd_alg in directories:
    data_training[pd_alg] = pd.DataFrame()
    label_training[pd_alg] = {}
    files = os.listdir(f"./cc_diagnoses/{pd_alg}/training/")
    for file in files:
        df = pd.read_csv(f"./cc_diagnoses/{pd_alg}/training/{file}", delimiter=',')
        data_training[pd_alg] = data_training[pd_alg].append(df, ignore_index=True)

    label_training[pd_alg] = data_training[pd_alg].iloc[:, -1].to_numpy()  # label
    data_training[pd_alg] = data_training[pd_alg].iloc[:, :-2].to_numpy()  # non prendo fitness e label
    label_training[pd_alg][label_training[pd_alg] == 'A'] = 1
    label_training[pd_alg][label_training[pd_alg] == 'N'] = 0
    label_training[pd_alg] = np.array(label_training[pd_alg].tolist())

# load test set
directories = os.listdir('./cc_diagnoses')
data_test = {}
label_test = {}
fitness = {}
output_parameters = {}
for pd_alg in directories:
    test_file = os.listdir(f"./cc_diagnoses/{pd_alg}/test")
    data_test[pd_alg] = pd.read_csv(f"./cc_diagnoses/{pd_alg}/test/{test_file[0]}", delimiter=',')

    label_test[pd_alg] = data_test[pd_alg].iloc[:, -1].to_numpy()
    fitness[pd_alg] = data_test[pd_alg].iloc[:, -2].to_numpy()
    data_test[pd_alg] = data_test[pd_alg].iloc[:, :-2].to_numpy()
    label_test[pd_alg][label_test[pd_alg] == 'A'] = 1
    label_test[pd_alg][label_test[pd_alg] == 'N'] = 0
    label_test[pd_alg] = np.array(label_test[pd_alg].tolist())

df = {}
for pd_alg in data_training.keys():
    print(pd_alg)
    df[pd_alg] = {}
    with open("parameters.json", "r") as file_:
        parameters = json.load(file_)
        for alg in parameters[pd_alg].keys():
            if alg == 'fitness':
                y_pred = fitness[pd_alg] < parameters[pd_alg][alg]['threshold']
                df[pd_alg]['fitness'] = [accuracy_score(label_test[pd_alg], y_pred),
                                      precision_score(label_test[pd_alg], y_pred),
                                      recall_score(label_test[pd_alg], y_pred), f1_score(label_test[pd_alg], y_pred)]
            if alg == 'RF':
                RF = RandomForestClassifier(**parameters[pd_alg][alg]['RF'])
                scaler = preprocessing.StandardScaler().fit(data_training[pd_alg])
                data_tr = scaler.transform(data_training[pd_alg])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[pd_alg][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)


                RF.fit(data_tr, label_training[pd_alg])
                data_ts = scaler.transform(data_test[pd_alg])
                data_ts = pca_opt.transform(data_ts)

                y_pred = RF.predict(data_ts)
                df[pd_alg]['RF'] = [accuracy_score(label_test[pd_alg], y_pred), precision_score(label_test[pd_alg], y_pred),
                                 recall_score(label_test[pd_alg], y_pred), f1_score(label_test[pd_alg], y_pred)]
            if alg == 'XGboost':
                XGB = xgb.XGBClassifier(**parameters[pd_alg][alg]['XGboost'])
                scaler = preprocessing.StandardScaler().fit(data_training[pd_alg])
                data_tr = scaler.transform(data_training[pd_alg])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[pd_alg][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                XGB.fit(data_tr, label_training[pd_alg])
                data_ts = scaler.transform(data_test[pd_alg])
                data_ts = pca_opt.transform(data_ts)

                y_pred = XGB.predict(data_ts)
                df[pd_alg]['XGboost'] = [accuracy_score(label_test[pd_alg], y_pred), precision_score(label_test[pd_alg], y_pred),
                                      recall_score(label_test[pd_alg], y_pred), f1_score(label_test[pd_alg], y_pred)]
            if alg == 'Adaboost':
                Ada = AdaBoostClassifier(**parameters[pd_alg][alg]['Adaboost'])
                scaler = preprocessing.StandardScaler().fit(data_training[pd_alg])
                data_tr = scaler.transform(data_training[pd_alg])

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[pd_alg][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                Ada.fit(data_tr, label_training[pd_alg])
                data_ts = scaler.transform(data_test[pd_alg])
                data_ts = pca_opt.transform(data_ts)

                y_pred = Ada.predict(data_ts)
                df[pd_alg]['Adaboost'] = [accuracy_score(label_test[pd_alg], y_pred),
                                       precision_score(label_test[pd_alg], y_pred), recall_score(label_test[pd_alg], y_pred),
                                       f1_score(label_test[pd_alg], y_pred)]
            if alg == 'SVM':
                SVM = OneClassSVM(**parameters[pd_alg][alg]['SVM'])
                normal_data = data_training[pd_alg][label_training[pd_alg] == 0]
                scaler = preprocessing.StandardScaler().fit(normal_data)
                data_tr = scaler.transform(normal_data)

                pca = PCA()
                X_new_train = pca.fit_transform(data_tr)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= parameters[pd_alg][alg]['pca']['variance_threshold']) + 1
                pca_opt = PCA(n_components=n_components)
                data_tr = pca_opt.fit_transform(data_tr)

                SVM.fit(data_tr)
                data_ts = scaler.transform(data_test[pd_alg])
                data_ts = pca_opt.transform(data_ts)

                y_pred = SVM.predict(data_ts)
                y_pred[y_pred == 1] = 0
                y_pred[y_pred == -1] = 1
                df[pd_alg]['SVM'] = [accuracy_score(label_test[pd_alg], y_pred), precision_score(label_test[pd_alg], y_pred),
                                  recall_score(label_test[pd_alg], y_pred), f1_score(label_test[pd_alg], y_pred)]
            if alg == 'IF':
                IF = IsolationForest(**parameters[pd_alg][alg]['IF'])
                scaler = preprocessing.StandardScaler().fit(data_training[pd_alg])
                data_tr = scaler.transform(data_training[pd_alg])


                IF.fit(data_tr)
                data_ts = scaler.transform(data_test[pd_alg])
                y_pred = IF.predict(data_ts)
                y_pred[y_pred == 1] = 0
                y_pred[y_pred == -1] = 1
                df[pd_alg]['IF'] = [accuracy_score(label_test[pd_alg], y_pred), precision_score(label_test[pd_alg], y_pred),
                                 recall_score(label_test[pd_alg], y_pred), f1_score(label_test[pd_alg], y_pred)]


for pd_alg in df.keys():
    with open(str(pd_alg) + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for alg in df[pd_alg].keys():
            writer.writerow([alg])
            writer.writerow(df[pd_alg][alg])
