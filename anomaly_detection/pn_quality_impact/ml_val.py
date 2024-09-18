import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import json
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from matplotlib.ticker import MaxNLocator
import copy
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.svm import OneClassSVM
import os
from sklearn.model_selection import GridSearchCV

class CustomPCA(PCA):
    def __init__(self, variance_threshold=None):
        super().__init__()
        self.variance_threshold = variance_threshold
        self.n_components_ = None

    def fit(self, X, y=None):
        super().fit(X, y)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        self.n_components_ = n_components
        self.n_components = n_components
        return self


directories = os.listdir('./cc_diagnoses')
output_parameters = {}
for pd_alg in directories:
        if pd_alg == 'ilp':
                print(pd_alg)
                output_parameters[pd_alg] = {}
                files = os.listdir(f"./cc_diagnoses/{pd_alg}/training/")
                data_training = pd.DataFrame()
                for file in files:
                        df = pd.read_csv(f"./cc_diagnoses/{pd_alg}/training/{file}", delimiter=',')
                        data_training = data_training.append(df, ignore_index=True)

                fitness = data_training.iloc[:, -2].to_numpy()  
                label_training = data_training.iloc[:, -1].to_numpy()
                data_training = data_training.iloc[:, :-2].to_numpy()
                label_training[label_training == 'A'] = 1
                label_training[label_training == 'N'] = 0
                label_training = np.array(label_training.tolist())


                fitness_nominal = fitness[label_training == 0]
                threshold = np.min(fitness_nominal)
                output_parameters[pd_alg]['fitness'] = {'threshold': threshold}


                # One class SVM
                svm_params_1 = [0.1, 0.2, 0.3] 
                svm_params_2 = ['linear', 'poly', 'rbf', 'sigmoid'] 
                svm_params_3 = [0.95, 0.96,0.97,0.98, 0.99]  
                f1 = {}
                acc_param = []
                acc_value = []
                for param_1 in svm_params_1:
                        for param_2 in svm_params_2:
                                for param_3 in svm_params_3:
                                        normal_data = data_training[label_training == 0]
                                        normal_label = label_training[label_training == 0]
                                        anomalous_data = data_training[label_training == 1]
                                        anomalous_label = label_training[label_training == 1]

                                        X_train = normal_data[0:int((len(normal_data)/10)*8)]

                                        scaler = preprocessing.StandardScaler().fit(X_train)
                                        X_train = scaler.transform(X_train)
                                        # scaling
                                        X_test_1 = normal_data[int((len(normal_data)/10)*8):-1]
                                        X_test_2 = anomalous_data
                                        X_test = np.append(X_test_1,X_test_2, axis=0)

                                        X_test = scaler.transform(X_test)

                                        y_test_1 = normal_label[int((len(normal_data)/10)*8):-1]
                                        y_test_2 = anomalous_label
                                        y_test = np.append(y_test_1, y_test_2, axis=0)

                                        # feature selection
                                        pca = PCA()
                                        X_new_train = pca.fit_transform(X_train)


                                        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                                        n_components = np.argmax(cumulative_variance >= param_3) + 1

                                        pca_opt = PCA(n_components=n_components)
                                        X_train = pca_opt.fit_transform(X_train)
                                        X_test = pca_opt.transform(X_test)


                                        SVM = OneClassSVM(nu=param_1, kernel=param_2)
                                        SVM.fit(X_train)
                                        y_pred = SVM.predict(X_test)

                                        y_test[y_test == 1] = -1
                                        y_test[y_test == 0] =  1
                                        cm = confusion_matrix(y_test, y_pred, labels=[1, -1])

                                        CM = np.transpose(cm)
                                        recall = CM[1][1] / sum(CM[:, 1])
                                        precision = CM[1][1] / sum(CM[1, :])
                                        f1[str({'nu': param_1, 'kernel': param_2})] = 2 * (precision * recall) / (precision + recall)
                                        acc_param.append({'SVM':{'nu': param_1, 'kernel': param_2},'pca':{'variance_threshold': param_3}})
                                        acc_value.append(2 * (precision * recall) / (precision + recall))


                output_parameters[pd_alg]['SVM'] = acc_param[acc_value.index(np.nanmax(acc_value))]

                list_params = []
                for i in range(len(acc_param)):
                        list_params.append(list(acc_param[i].values()))
                fig = plt.figure(figsize=(19.20, 10.80), tight_layout=True)
                ax = plt.subplot(111)
                for i in range(len(acc_param)):
                        plt.bar("(" + str(list(acc_param[i].values())[0]['nu']) + ", " + str(list(acc_param[i].values())[0]['kernel']) + ", " + str(list(acc_param[i].values())[1]['variance_threshold']) + ")", acc_value[i]*100, hatch='o', edgecolor='black', linewidth=0.5, color='grey')
                plt.xticks(rotation=35, ha='right', fontsize=11, fontweight='bold')
                plt.yticks(fontsize=25, fontweight='bold')
                ax.yaxis.set_major_locator(MaxNLocator(nbins=20))
                ax.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
                plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xlabel(r"Hyperparameter Configurations ($\bf{\nu}$, kernel, variance threshold)", fontweight="bold", fontsize=30)
                plt.ylabel('F1 [%]', fontweight="bold", fontsize=30)
                print(acc_value)
                plt.savefig('val_ilp_25_ocsvm.png', format="png", dpi=600, bbox_inches='tight')

                #XGboost
                pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', CustomPCA()),
                        ('model', xgb.XGBClassifier())
                ])
                xgb_params = {'model__booster': ['gbtree', 'gblinear'], 'model__eta': [0.001, 0.01, 0.1, 0.3, 0.5, 1], 'model__n_estimators': [100,200,500,1000], 'pca__variance_threshold': [0.95, 0.96,0.97,0.98, 0.99]}
                clf = GridSearchCV(pipeline, xgb_params, cv=5, scoring='accuracy')
                clf.fit(data_training, label_training)
                print("XGboost")
                print(clf.best_params_)
                print(clf.best_score_)
                best_param = copy.deepcopy(clf.best_params_)
                best_param['booster'] = best_param['model__booster']
                del best_param['model__booster']
                best_param['eta'] = best_param['model__eta']
                del best_param['model__eta']
                best_param['n_estimators'] = best_param['model__n_estimators']
                del best_param['model__n_estimators']
                best_param['variance_threshold'] = best_param['pca__variance_threshold']
                del best_param['pca__variance_threshold']
                output_parameters[pd_alg]["XGboost"] = {'XGboost':{'booster': best_param['booster'],'eta': best_param['eta'],'n_estimators': best_param['n_estimators']}, 'pca': {'variance_threshold':best_param['variance_threshold']}}
                for i in range(len(clf.cv_results_['mean_test_score'])):
                        plt.bar(f"{clf.cv_results_['params'][i]['model__booster']} {clf.cv_results_['params'][i]['model__eta']} {clf.cv_results_['params'][i]['model__n_estimators']}", clf.cv_results_['mean_test_score'][i], color='gray')
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.show()



                #Random Forest
                pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', CustomPCA()),
                        ('model', RandomForestClassifier())
                ])
                forest_params = [{'model__n_estimators': [50,100,200,500,1000], 'pca__variance_threshold': [0.95, 0.96,0.97,0.98, 0.99]}]
                clf = GridSearchCV(pipeline, forest_params, cv=5, scoring='accuracy')
                clf.fit(data_training, label_training)
                best_param = copy.deepcopy(clf.best_params_)
                best_param['n_estimators'] = best_param['model__n_estimators']
                del best_param['model__n_estimators']
                best_param['variance_threshold'] = best_param['pca__variance_threshold']
                del best_param['pca__variance_threshold']
                output_parameters[pd_alg]["RF"] = {'RF':{'n_estimators': best_param['n_estimators']}, 'pca':{'variance_threshold':  best_param['variance_threshold']}}
                for i in range(len(clf.cv_results_['mean_test_score'])):
                        plt.bar(str(clf.cv_results_['params'][i]['model__n_estimators']), clf.cv_results_['mean_test_score'][i], color='gray')

                        
                #Adaboost
                pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', CustomPCA()),
                        ('model', AdaBoostClassifier())
                ])
                ada_params = {'model__n_estimators': [100,200,500,1000], 'model__learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5, 1], 'pca__variance_threshold': [0.95, 0.96,0.97,0.98, 0.99]}
                clf = GridSearchCV(pipeline, ada_params, cv=5, scoring='accuracy')
                clf.fit(data_training, label_training)
                best_param = copy.deepcopy(clf.best_params_)
                best_param['n_estimators'] = best_param['model__n_estimators']
                del best_param['model__n_estimators']
                best_param['learning_rate'] = best_param['model__learning_rate']
                del best_param['model__learning_rate']
                best_param['variance_threshold'] = best_param['pca__variance_threshold']
                del best_param['pca__variance_threshold']

                for i in range(len(clf.cv_results_['mean_test_score'])):
                        plt.bar(f"{clf.cv_results_['params'][i]['model__n_estimators']} {clf.cv_results_['params'][i]['model__learning_rate']}", clf.cv_results_['mean_test_score'][i], color='gray')
                plt.xticks(rotation=45, ha='right', fontsize=10)
                output_parameters[pd_alg]["Adaboost"] = {'Adaboost':{'n_estimators': best_param['n_estimators'],'learning_rate': best_param['learning_rate']}, 'pca':{'variance_threshold': best_param['variance_threshold']}}

                # Isolation Forest
                params_1 = [100,200,500] 
                params_2 = [0.1,0.2,0.3,0.4]
                params_3 = [0.95, 0.96,0.97,0.98, 0.99]
                f1 = {}
                acc_param = []
                acc_value = []
                for param_1 in params_1:
                        for param_2 in params_2:
                                for param_3 in params_3:
                                        CM = [[0, 0], [0, 0]]
                                        kf = StratifiedKFold(n_splits=5, shuffle=True)
                                        for train_index, test_index in kf.split(data_training, label_training):
                                                X_train = data_training[train_index]
                                                # scaling
                                                scaler = preprocessing.StandardScaler().fit(X_train)
                                                X_train = scaler.transform(X_train)
                                                y_train = label_training[train_index]
                                                X_test = data_training[test_index]
                                                X_test = scaler.transform(X_test)
                                                y_test = label_training[test_index]

                                                # feature selection
                                                pca = PCA()
                                                X_new_train = pca.fit_transform(X_train)


                                                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                                                n_components = np.argmax(cumulative_variance >= param_3) + 1

                                                pca_opt = PCA(n_components=n_components)
                                                X_train = pca_opt.fit_transform(X_train)
                                                X_test = pca_opt.transform(X_test)

                                                IF = IsolationForest(n_estimators = param_1, contamination=param_2, max_samples = 'auto')
                                                IF.fit(X_train)

                                                best_param['n_components'] = n_components

                                                y_pred = IF.predict(X_test)
                                                y_test[y_test == 1] = -1
                                                y_test[y_test == 0] = 1
                                                cm = confusion_matrix(y_test, y_pred,
                                                                      labels=[1, -1]) 

                                                for i in range(len(cm)):
                                                        for j in range(len(cm[0])):
                                                                CM[i][j] = CM[i][j] + cm[i][j]

                                        CM = np.transpose(cm)
                                        recall = CM[1][1] / sum(CM[:, 1])
                                        precision = CM[1][1] / sum(CM[1, :])
                                        acc_param.append({'IF':{'n_estimators': param_1,'contamination':param_2},'pca':{'variance_threshold':param_3}})
                                        acc_value.append(np.trace(CM) / sum(CM))

                output_parameters[pd_alg]['IF'] = acc_param[acc_value.index(np.nanmax(acc_value))]

                list_params = []
                for i in range(len(acc_param)):
                        list_params.append(list(acc_param[i].values()))
                for i in range(len(acc_param)):
                        plt.bar(str(list(acc_param[i].values())[0]) + " " + str(list(acc_param[i].values())[1]), acc_value[i], color='blue')

with open(f"./parameters.json", "w") as file:
    json.dump(output_parameters, file)
