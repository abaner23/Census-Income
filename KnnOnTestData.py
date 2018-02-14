

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:44:25 2018

@author: barna
"""

import pandas as pd
import numpy as np
import math as mt
import sklearn.metrics as met
def MapWithOrdinal(val1,val2):
    out1=0
    out2=0
    if(val1>=1 and val1<=8):
       out1=1
    elif(val1>=9 and val1<=12):
       out1=2
    elif(val1==13):
        out1=6
    elif(val1==14 or val1==15):
        out1=12
    elif(val1==16):
        out1=24
        
    if(val2>=1 and val2<=8):
       out2=1
    elif(val2>=9 and val2<=12):
       out2=2
    elif(val2==13):
        out2=6
    elif(val2==14 or val2==15):
        out2=12
    elif(val2==16):
        out2=24
    return [out1,out2]
def CustomDissimilarityFunction(A,B,nominal,ordinal,continuous,variances):
    #Relationship, Education deleted
    #Gender- Male or female If A[gender]=B[gender] then 0 othewise 1
    #Native- Countries If A[country]=B[country] then 0 otherwise 1
    #Marital Status same, Race same
    #Age = |A[age]-B[age]| hours per week same capital gain and capital loss same, finalwgt same
    #Education- ordinal variable
        #1-8 education level- 1
        #9-12 education level- 2
        #13- 6
        #14-15- 12
        #16- 24
    d_nominal=0
    d_ordinal=0
    d_conti=0
    for i in range(0,nominal.shape[0]):
        if(A[nominal[i]]!=B[nominal[i]]):
            d_nominal=d_nominal+1
    for i in range(0,ordinal.shape[0]):
        [mapped_A, mapped_B]=MapWithOrdinal(A[ordinal[i]],B[ordinal[i]])
        d_ordinal=d_ordinal+abs((mapped_A-mapped_B)/(24-1))
    for i in range(0,continuous.shape[0]):
        d_conti=d_conti+(abs(A[continuous[i]]-B[continuous[i]])/mt.sqrt(variances[i]))
    return d_nominal+d_ordinal+d_conti

def tocat(h):
    #print(type(h))
    if(h==' >50K'):
        return 1
    else:
        return 0
    
def PrintKNearestNeighbors(filelocation,k=5,threshold=0.5):
    
    trainset=pd.read_csv("C:\\Users\\barna\\Downloads\\income_tr.csv")
    trainset['class'] = trainset['class'].apply(tocat)
    testset=pd.read_csv(filelocation)
    testset['class'] = testset['class'].apply(tocat)
    testset_classes=testset.iloc[:,15]
    trainset_classes=trainset.iloc[:,15]
    testset.drop(['ID','education','relationship','class'],axis=1,inplace=True)
    trainset.drop(['ID','education','relationship','class'],axis=1,inplace=True)
    cols = testset.columns
    continuous_variables= np.array(testset._get_numeric_data().columns)
    nominal_variables=np.array(list(set(cols) - set(continuous_variables)))
    ordinal_variables=np.array([cols[3]])
    continuous_variables=np.delete(continuous_variables,[2])
    variances=np.empty(continuous_variables.shape[0],dtype=float)
    
    for i in range(0,continuous_variables.shape[0]):
        variances[i]=np.var(np.array(trainset[continuous_variables[i]]))
        
    DissimilarityIndex=np.empty([testset.shape[0],trainset.shape[0]],dtype=float)
    for i in range(0,testset.shape[0]):
        for j in range(0, trainset.shape[0]):
            DissimilarityIndex[i,j]=-1
    
    range1=50
    predicted_classes=np.empty(range1,dtype=int)
    
    for i in range(0,range1):
        for j in range(0, trainset.shape[0]):
            if(i!=j and DissimilarityIndex[i,j]==-1):
                DissimilarityIndex[i,j]=CustomDissimilarityFunction(testset.iloc[i,:],trainset.iloc[j,:],nominal_variables,ordinal_variables,continuous_variables,variances)
        closest_indexes=np.array(DissimilarityIndex[i,:].argsort()[:k+1],dtype=int)
        classes_closest=np.array(trainset_classes[closest_indexes[1:]])
        
        number_classes1=np.sum(classes_closest)
       
        number_classes0=k-number_classes1
        if((number_classes1/k)>=threshold):
            predicted_class=1
            posterior=number_classes1/k
        else:
            predicted_class=0
            posterior=number_classes0/k
        predicted_classes[i]=predicted_class
        print("{} {} {} {}".format(i,testset_classes[i],predicted_class,posterior))

    #This section gives the model performance
    
    confusion_m=met.confusion_matrix(testset_classes[0:range1],predicted_classes)
    print(confusion_m)
    print("Accuracy={}".format((confusion_m[0,0]+confusion_m[1,1])/range1))
    print("Error rate={}".format(1.0-(confusion_m[0,0]+confusion_m[1,1])/range1))
    print("True Negative={}".format(confusion_m[0,0]))
    print("True Positive={}".format(confusion_m[1][1]))
    print("False Positive={}".format(confusion_m[0][1]))
    print("False Negative={}".format(confusion_m[1][0]))
    print("Precision={}".format(confusion_m[1][1]/(confusion_m[1][1]+confusion_m[0][1])))
    print("Recall={}".format(confusion_m[1][1]/(confusion_m[1][1]+confusion_m[1][0])))
    print("F-score={}".format((2*confusion_m[1][1])/((confusion_m[1][1]+confusion_m[0][1])+(confusion_m[1][1]+confusion_m[1][0]))))

def KnnOnTestDataset(filelocation,k=5,threshold=0.5):
    
    if(k%2==0):
        print('Please pass an odd value of k')
    else:
        PrintKNearestNeighbors(filelocation,k,threshold)
            
    return

KnnOnTestDataset("C:\\Users\\barna\\Downloads\\income_te.csv",5,0.62)
