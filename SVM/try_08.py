#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:40:54 2018

@author: pepe
"""

import numpy as np
from sklearn.svm import SVC
import cv2
from scipy import signal
import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from imutils.object_detection import non_max_suppression
import math
import scipy.misc
from skimage.feature import hog
import os
import glob
from sklearn.externals import joblib
 
hogcv2 = cv2.HOGDescriptor()
svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                   svm_type = cv2.ml.SVM_C_SVC,
                   C=2.67, 
                   gamma=5.383 )

def hoggify(path,extension,is_color):
    data=[]
    lista=glob.glob(os.path.join(path,"*"+extension))
    lista=np.squeeze(lista)
    for file in lista:
        image = cv2.imread(file, is_color)
#        dim=256
#        dim = 128
        dim = 128
        img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
        img = hogcv2.compute(img)
        img = np.squeeze(img)
        data.append(img)
    return data

def hoggify2(path,extension,is_color):
    data=[]
    lista=glob.glob(os.path.join(path,"*"+extension))
    lista=np.squeeze(lista)
    for file in lista:
        image = cv2.imread(file, is_color)
        dim = 128
        img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog2 = cv2.HOGDescriptor(winSize,blockSize,blockStride,
                            cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        img = hog2.compute(img)
        img = np.squeeze(img)
        data.append(img)
    return data

def svm_classify(features,labels):
    clf=SVC(C=10000,kernel="linear",gamma=0.000001,probability=True)
    clf.fit(features,labels)
    joblib.dump(clf, 'modelo_svm_scikit.joblib')
    return clf

def svm_classify2(features,labels):
    svm=cv2.ml.SVM_create()
    svm.train(trainData,responses, params=svm_params)
    svm.save('modelo_svm_cv2.dat')
    return svm

def list_to_matrix(lst):
    return np.stack(lst) 

def s_x(img):
    kernel = np.array([[-1, 0, 1]])
    imgx = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    return imgx

def s_y(img):
    kernel = np.array([[-1, 0, 1]]).T
    imgy = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    return imgy

def grad(img):
    imgx = s_x(img)
    imgy = s_y(img)
    s = np.sqrt(imgx**2 + imgy**2)
    theta = np.arctan2(imgx, imgy)
    theta[theta<0] = np.pi + theta[theta<0]
    return (s, theta)

PATH_POSITIVAS = "/home/pepe/DATOS/imagenes/caras/s1"
PATH_NEGATIVAS = "/home/pepe/DATOS/imagenes/caras/s2"
MODEL_FILENAME = "./model"

if __name__=='__main__':
    dataHogPositivos=hoggify(PATH_POSITIVAS,'pgm',False)
    dataHogNegativos=hoggify(PATH_NEGATIVAS,'pgm',False)
    n=len(dataHogPositivos)
    m=len(dataHogNegativos)
    dataHog=dataHogPositivos+dataHogNegativos
    labelsHog=[0]*n+[1]*m
    clf=svm_classify(dataHog,labelsHog)
    X_test=dataHogPositivos[-1:]
    salida=clf.predict(X_test)
    probability=clf.predict_proba(X_test) # aqui calculo las probabilidades
    log_probabilities=clf.predict_log_proba(X_test)
    print("probabilidad 0: %f , 1:%f"%(math.exp(log_probabilities[0,0]),math.exp(log_probabilities[0,1])))
    X_test=dataHogNegativos[-1:]
    salida=clf.predict(X_test)
    probability=clf.predict_proba(X_test) # aqui calculo las probabilidades
    log_probabilities=clf.predict_log_proba(X_test)
    print("probabilidad 0: %f , 1:%f"%(math.exp(log_probabilities[0,0]),math.exp(log_probabilities[0,1])))