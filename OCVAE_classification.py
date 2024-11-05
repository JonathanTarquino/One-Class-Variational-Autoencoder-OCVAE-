import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as myimg
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn import preprocessing
from sklearn import svm
import pandas as pd
import glob
import cv2 as cv
from sklearn.metrics import roc_curve, roc_auc_score


random_state = 12883823
rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=random_state)
KNN_accuracy = []
SVM_accuracy = []
f1_score_result = []
clf_recall_score = []
RDF_accuracy = []
grid_accuracy = []
auc_svm = []
grid_auc =[]
prediction = pd.DataFrame([])



# Reading the VAE latent space
T0 = pd.read_csv('/home/jonathan/Documentos/UNAL/Thyroid/code/data_test_means1.csv')#d5_q64, d7_q32, d7_q16
#T0 = pd.read_csv('/home/jonathan/Documentos/UNAL/Leucemia/VAE_means_variances.csv')
T0 = T0.dropna(axis = 'index')

data_tirads = T0.iloc[:,1:65]
print(data_tirads)
target = np.array(T0.iloc[:,65])
print(np.shape(target),target)
#test = data_tirads.iloc[0:1,:]
balanced = []
count=0
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(data_tirads)

##################################################################################
##  Following part is for data spliting and classfication validation           ###
##  train: contains a data subset for classifiers trainning (just indexes)     ###
##  test:  contains a data subset for classifiers testing (just indexes)       ###
##        K-nn, Random Forest and SVM classifiers were used                    ###
##################################################################################

for train, test in rkf.split(scaled_data):
    #print(data.iloc[train])
    # Spliting data for training and testing
    X_train,X_test, y_train, y_test = scaled_data[train],scaled_data[test], target[train], target[test]
    #X_train = preprocessing.scale(X_train)
    #X_test = preprocessing.scale(X_test)
    
    #### Data follow up table: Saving the position of data used for testing and its associated labels
    #result = np.transpose([test,y_test]) 
    #prediction = pd.concat([prediction, pd.DataFrame(result)], axis = 1)
    print('Test set:\n',np.shape(X_test))

    # Defining KNN parameter for classification task
    KNN_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    KNN_prediction=KNN_model.predict(X_test) #binary prediction
    KNN_prediction_proba = KNN_model.predict_proba(X_test) # prediction probability
    KNN_accuracy = [np.append(KNN_accuracy, accuracy_score(KNN_prediction, y_test))]


    fpr_knn, tpr_knn, _ = roc_curve(y_test, KNN_prediction_proba[:,1])
    auc_knn =roc_auc_score(y_test,KNN_prediction)
    #plt.plot(fpr_knn, tpr_knn, label='KNN:'+str(auc_knn))


    # SVM CLASSIFIER
    clf_model = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
    clf_prediction=clf_model.predict(X_test)
    y = clf_model.decision_function(X_test)
    SVM_accuracy = [np.append(SVM_accuracy, accuracy_score(clf_prediction, y_test))]
    f1_score_result =[np.append(f1_score_result,f1_score(clf_prediction,y_test))]
    clf_recall_score =[np.append(clf_recall_score,recall_score(clf_prediction,y_test))]
    print(classification_report(y_test, clf_prediction))

    print('Linear accuracy.....................................',SVM_accuracy)
    
    ## This section is for estimating (w-x)^2 = ww'+ xx'-wx as the distance btween observations and suport vectors
    print('Support vectors:\n',np.shape(clf_model.support_vectors_))
    sv = clf_model.support_vectors_
    
    clf_prediction_proba = clf_model.predict_proba(X_test) # prediction probability
    fpr_svm, tpr_svm, _ = roc_curve(y_test, clf_prediction_proba[:,1])


    auc_svm = [np.append(auc_svm,roc_auc_score(y_test,clf_prediction))]
    #plt.plot(fpr_svm, tpr_svm, label='SVM rbf k:'+str(auc_svm))
    #scores = cross_val_score(clf_model, data, target, cv=2)
    #print('SVM accuracy:',scores)
    
    ####################################################################################
    ##   GRID SVM OPTIMIZATION by varying 'C' constrain and 'gamma' using RBF kernel ###
    ####################################################################################
    
    #param_grid = {'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['linear']}  
    #grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
    ## fitting the model for grid search 
    #grid.fit(X_train, y_train) 
    
    
    #grid_predictions = grid.predict(X_test) 
    #y = grid.decision_function(X_test)
    
  
    ## print classification report 
    ##print(classification_report(y_test, grid_predictions)) 
    #grid_auc = [np.append(grid_auc,roc_auc_score(y_test,grid_predictions))]
    #grid_accuracy = [np.append(grid_accuracy, accuracy_score(grid_predictions, y_test))]
    
    ##### Adding classifier prediction to data follow up table
    #prediction = pd.concat([prediction, pd.DataFrame(clf_prediction),pd.DataFrame(y / np.linalg.norm(clf_model.coef_))], axis = 1)
    #print('Prediction:\n',prediction)
    #############################################################################


    rfc_model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    rfc_prediction = rfc_model.predict(X_test)
    RDF_accuracy = [np.append(RDF_accuracy, accuracy_score(rfc_prediction, y_test))]

    #scores = cross_val_score(rfc_model, data, target, cv=5)
    #print(np.mean(scores))
    rfc_prediction_proba = rfc_model.predict_proba(X_test) # prediction probability
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, rfc_prediction_proba[:,1])
    auc_rf=roc_auc_score(y_test,rfc_prediction)
    #plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF:'+str(auc_rf))


prediction.to_csv(r'/home/jonathan/Documentos/UNAL/Leucemia/Imagenes_marcadas/ALL_IDB2/predictions.csv', header=None)

print('Random forest Accuracy:',np.mean(RDF_accuracy), np.std(RDF_accuracy))    
print('SVM Accuracy:',np.mean(SVM_accuracy), np.std(SVM_accuracy))
print('SVM AUC:',np.mean(auc_svm), np.std(auc_svm))
print('SVM recall:',np.mean(clf_recall_score), np.std(clf_recall_score))
print('F1 score:', np.mean(f1_score_result),np.std(f1_score_result))

