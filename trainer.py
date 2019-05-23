import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
import os
import glob
import re
import datetime
import math
import random

nFeatures = 12
NCV = 10
nClasses = 2

def printStats(dataPath):
    file_string = dataPath + '/training/'
    ytrain=np.load(file_string + 'ytrain0.dat')
    yval=np.load(file_string + 'ytest0.dat')
    nPos = (ytrain == 1).sum() + (yval == 1).sum()
    nNeg = (ytrain == 0).sum() + (yval == 0).sum()
    print 'Positive (flaring AR) training samples = %d'%nPos
    print 'Negative (nonflaring AR) training samples = %d'%nNeg
    return    

def calcTSS(confmat):
    acc = np.empty(nClasses,dtype=np.float32)
    rec = np.empty(nClasses,dtype=np.float32)
    tss = np.empty(nClasses,dtype=np.float32)
    for j in xrange(nClasses):
        tp=float(confmat[j,j])
        tn=0.0
        for k in xrange(j+1,nClasses):
            tn += confmat[k,k]
        for k in xrange(0,j):
            tn += confmat[k,k]
        fp=confmat[:,j].sum()-tp
        fn=confmat[j,:].sum()-tp
        acc[j] = ( tp + tn ) / ( tp + tn + fp + fn )
        rec[j] = ( tp ) / ( tp + fn )
        tss[j] = ( tp ) / ( tp + fn ) - ( fp ) / ( fp + tn )
    return acc,rec,tss

def LogitClassifier(dataPath, C=0.01, W=6):   
    file_string = dataPath + '/training/'
    acc = np.empty((NCV,nClasses),dtype=np.float32)
    recall = np.empty((NCV,nClasses),dtype=np.float32) 
    tss = np.empty((NCV,nClasses),dtype=np.float32)
    for loop in xrange(NCV):
        xtrain=np.load(file_string + 'xtrain%d.dat'%(loop))
        ytrain=np.load(file_string + 'ytrain%d.dat'%(loop))
        
        xval=np.load(file_string + 'xtest%d.dat'%(loop))
        yval=np.load(file_string + 'ytest%d.dat'%(loop))
        
        xtrain = np.array(xtrain,dtype=np.float)
        xval = np.array(xval,dtype=np.float) 
        
        clf=LogisticRegression(penalty = 'l2',C=C,class_weight={1:W},verbose=0)
        clf.fit(xtrain,ytrain)
        ytpred=clf.predict(xtrain)
        yvpred=clf.predict(xval)
        confmat=confusion_matrix(yval,yvpred)
        acc[loop], recall[loop], tss[loop] = calcTSS(confmat)
    print '--------------------------Logistic Regression classification--------------------------'
    print r'Accuracy = %0.4f +/- %0.04f'%(acc[:,0].mean(),acc[:,0].std()) 
    print r'Positive class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,1].mean(),recall[:,1].std()) 
    print r'Negative class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,0].mean(),recall[:,0].std()) 
    print r'TSS = %0.4f +/- %0.04f'%(tss[:,0].mean(),tss[:,0].std()) 
    
    return 

def overSampler(dataPath, factor=4, valFlag=0):
    osdataPath = dataPath + '/training/os'
    if not os.path.exists(osdataPath):
        os.makedirs(osdataPath)
    for cv in xrange(NCV):
        xtrain = np.load(dataPath + '/training/xtrain%d.dat'%(cv))
        xtest = np.load(dataPath + '/training/xtest%d.dat'%(cv))
        ytrain = np.load(dataPath + '/training/ytrain%d.dat'%(cv))
        ytest = np.load(dataPath + '/training/ytest%d.dat'%(cv))
       
        posIds = np.argwhere(ytrain == 1)[:,0]
        xtrain = list(xtrain)
        ytrain = list(ytrain)
        for r in xrange(factor):
            for id in posIds:
                xtrain.append(xtrain[id])
                ytrain.append(ytrain[id])
        xtrain = np.array(xtrain, dtype=np.float32)
        ytrain = np.array(ytrain, dtype=np.int32)
        
        if valFlag:
            posIds = np.argwhere(ytest == 1)[:,0]
            xtest = list(xtest)
            ytest = list(ytest)
            for r in xrange(factor):
                for id in posIds:
                    xtest.append(xtest[id])
                    ytest.append(ytest[id])
            xtest = np.array(xtest, dtype=np.float32)
            ytest = np.array(ytest, dtype=np.int32)
        
        xtrain.dump(osdataPath + '/xtrain%d.dat'%(cv))
        xtest.dump(osdataPath + '/xtest%d.dat'%(cv))
        ytrain.dump(osdataPath + '/ytrain%d.dat'%(cv))
        ytest.dump(osdataPath + '/ytest%d.dat'%(cv))
    return


def GBclassifier(dataPath,lr=0.05,factor=4):   
    overSampler(dataPath, factor=factor)
    
    n_tree = 25
    max_depth = 4
    min_samples_split = 500
    min_samples_leaf = 15
    num_features = 10
    
    file_string = dataPath + '/training/os/' 
    
    acc = np.empty((NCV,nClasses),dtype=np.float32)
    recall = np.empty((NCV,nClasses),dtype=np.float32) 
    tss = np.empty((NCV,nClasses),dtype=np.float32)
    for loop in xrange(NCV):
        xtrain=np.load(file_string + 'xtrain%d.dat'%(loop))
        ytrain=np.load(file_string + 'ytrain%d.dat'%(loop))
        
        xval=np.load(file_string + 'xtest%d.dat'%(loop))
        yval=np.load(file_string + 'ytest%d.dat'%(loop))
        
        xtrain = np.array(xtrain,dtype=np.float)
        xval = np.array(xval,dtype=np.float) 
        
        clf=GradientBoostingClassifier(learning_rate = lr, n_estimators = n_tree, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = num_features, subsample = 0.8,verbose=0)
        clf.fit(xtrain,ytrain)
        ytpred=clf.predict(xtrain)
        yvpred=clf.predict(xval)
        confmat=confusion_matrix(yval,yvpred)
        acc[loop], recall[loop], tss[loop] = calcTSS(confmat)
    print '--------------------------Gradient Boosting classification--------------------------'
    print r'Accuracy = %0.4f +/- %0.04f'%(acc[:,0].mean(),acc[:,0].std()) 
    print r'Positive class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,1].mean(),recall[:,1].std()) 
    print r'Negative class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,0].mean(),recall[:,0].std()) 
    print r'TSS = %0.4f +/- %0.04f'%(tss[:,0].mean(),tss[:,0].std()) 
    
    return    

def SVMclassifier(dataPath,C=3.0,G=0.01,W=7.0):
    file_string = dataPath + '/training/'
    acc = np.empty((NCV,nClasses),dtype=np.float32)
    recall = np.empty((NCV,nClasses),dtype=np.float32) 
    tss = np.empty((NCV,nClasses),dtype=np.float32)
    for loop in xrange(NCV):
        xtrain=np.load(file_string + 'xtrain%d.dat'%(loop))
        ytrain=np.load(file_string + 'ytrain%d.dat'%(loop))
        
        xval=np.load(file_string + 'xtest%d.dat'%(loop))
        yval=np.load(file_string + 'ytest%d.dat'%(loop))
        
        xtrain = np.array(xtrain,dtype=np.float)
        xval = np.array(xval,dtype=np.float) 
        
        clf=svm.SVC(gamma=G,C=C,verbose=0,class_weight={1:W})
        clf.fit(xtrain,ytrain)
        ytpred=clf.predict(xtrain)
        yvpred=clf.predict(xval)
        confmat=confusion_matrix(yval,yvpred)
        acc[loop], recall[loop], tss[loop] = calcTSS(confmat)
    print '--------------------------SVM classification--------------------------'
    print r'Accuracy = %0.4f +/- %0.04f'%(acc[:,0].mean(),acc[:,0].std()) 
    print r'Positive class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,1].mean(),recall[:,1].std()) 
    print r'Negative class (flaring AR) recall %0.4f +/- %0.04f'%(recall[:,0].mean(),recall[:,0].std()) 
    print r'TSS = %0.4f +/- %0.04f'%(tss[:,0].mean(),tss[:,0].std()) 
    
    return 

def getClassifier(dataPath,machine_type='SVM'):
    clfs = {}
    clfs['SVM'] = svm.SVC(gamma=0.01,C=3.0,verbose=0,class_weight={1:7}) 
    clfs['Logit'] = LogisticRegression(penalty = 'l2',C=0.01,class_weight={1:6},verbose=0)
    n_tree = 25
    max_depth = 4
    min_samples_split = 500
    min_samples_leaf = 15
    num_features = 10
    clfs['GB'] = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = n_tree, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = num_features, subsample = 0.8,verbose=0)
    file_string = dataPath + '/training/'
    if machine_type == 'GB':
        file_string += 'os/'
        overSampler(dataPath, factor=4, valFlag=1)
    xtrain=np.load(file_string + 'xtrain0.dat')
    ytrain=np.load(file_string + 'ytrain0.dat')
    xval=np.load(file_string + 'xtest0.dat')
    yval=np.load(file_string + 'ytest0.dat')
    allXtrain = np.append(xtrain,xval,axis=0)
    allYtrain = np.append(ytrain,yval,axis=0)
    clfs[machine_type].fit(allXtrain,allYtrain)
    
    return clfs[machine_type]
    