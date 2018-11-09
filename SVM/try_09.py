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
# import svm as svmLibsvm
import pprint
import pickle
import yaml
import re
#local libraries
#from . import pputils
import pputils
#from libsvm.svm import *
#from libsvm.svmutil import *
#import cPickle


#PATH_POSITIVAS = "/home/pepe/DATOS/imagenes/caras/s1"
#PATH_NEGATIVAS = "/home/pepe/DATOS/imagenes/caras/s2"
PATH_POSITIVAS = "/home/pepe/DATOS/imagenes/MuestrasPlayers/Humanos"
PATH_NEGATIVAS = "/home/pepe/DATOS/imagenes/MuestrasPlayers/NoHumanos"
#PATH_POSITIVAS_TRAIN = "/home/pepe/DATOS/imagenes/MuestrasPlayers/HumanosTrain"
#PATH_NEGATIVAS_TRAIN = "/home/pepe/DATOS/imagenes/MuestrasPlayers/NoHumanosTrain"
# PATH_POSITIVAS_TEST = "/home/pepe/DATOS/imagenes/MuestrasPlayers/HumanosTest"
# PATH_NEGATIVAS_TEST = "/home/pepe/DATOS/imagenes/MuestrasPlayers/NoHumanosTest"
PATH_POSITIVAS_TRAIN = "/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTrain"
PATH_NEGATIVAS_TRAIN = "/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTrain"
PATH_POSITIVAS_TEST = "/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTest"
PATH_NEGATIVAS_TEST = "/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTest"
EXTENSION = 'png'
MODEL_PATH= "/home/pepe/DATOS/imagenes/MuestrasPlayers4"
MODEL_FILENAME = "./model"

 
hogcv2 = cv2.cv2.HOGDescriptor()
svm_params = dict( kernel_type = cv2.cv2.ml.SVM_LINEAR,
                   svm_type = cv2.cv2.ml.SVM_C_SVC,
                   C=2.67, 
                   gamma=5.383 )

# This function extract features indepently of size of image
def hoggify(path,extension,is_color):
    data=[]
    lista=glob.glob(os.path.join(path,"*"+extension))
    lista=np.squeeze(lista)
    for file in lista:
        image = cv2.cv2.imread(file, is_color)
        dim = 128
        img = cv2.cv2.resize(image, (dim,dim), interpolation = cv2.cv2.INTER_AREA)
        img = hogcv2.compute(img)
        img = np.squeeze(img)
        data.append(img)
    return data

def hoggify_folder(path,extension,is_color,listOfFiles):
    data=[]
    lista=glob.glob(os.path.join(path,"*"+extension))
    lista=np.squeeze(lista)
    for file in lista:
        listOfFiles.append(os.path.basename(file))
        image = cv2.cv2.imread(file, is_color)
        dim = 128
        img = cv2.cv2.resize(image, (dim,dim), interpolation = cv2.cv2.INTER_AREA)
        img = hogcv2.compute(img)
        img = np.squeeze(img)
        data.append(img)
    return data

def hoggify_image(image_file_name,extension,is_color):
    data=[]
    file=image_file_name+'.'+extension
    image = cv2.cv2.imread(file, is_color)
    dim = 128
    img = cv2.cv2.resize(image, (dim,dim), interpolation = cv2.cv2.INTER_AREA)
    img = hogcv2.compute(img)
    img = np.squeeze(img)
    data.append(img)
    return data

def hoggify2(path,extension,is_color):
    data=[]
    lista=glob.glob(os.path.join(path,"*"+extension))
    lista=np.squeeze(lista)
    for file in lista:
        image = cv2.cv2.imread(file, is_color)
        dim = 128
        img = cv2.cv2.resize(image, (dim,dim), interpolation = cv2.cv2.INTER_AREA)
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
        hog2 = cv2.cv2.HOGDescriptor(winSize,blockSize,blockStride,
                            cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        img = hog2.compute(img)
        img = np.squeeze(img)
        data.append(img)
    return data

def svm_classify(features,labels):
    clf=SVC(C=10000,kernel="linear",gamma=0.000001,probability=True)
    clf.fit(features,labels)
    joblib.dump(clf, MODEL_PATH+'/'+'modelo_svm_scikit.joblib')
    with open(MODEL_PATH+'/'+'modelo_svm_scikit.pkl', 'wb') as fileoutput:
        pickle.dump(clf, fileoutput)
    return clf

# Hasta donde he revisado no es posible obtener la probabilidad usando svm de opencv
def svm_classify2(features,labels):
    svm2 = cv2.cv2.ml.SVM_create()
    svm2.setKernel(cv2.cv2.ml.SVM_LINEAR)
    svm2.setType(cv2.cv2.ml.SVM_C_SVC)
    svm2.setC(2.67)
    svm2.setP(0.2)
#    svm.train(trainData,responses, params=svm_params)
    responses = np.array(labels).flatten()
    trainData = np.array(features)
#    svm2.train(trainData,responses, params=svm_params)
    svm2.train(trainData,cv2.cv2.ml.ROW_SAMPLE,responses)
#    svm2.train(features, cv2.ml.ROW_SAMPLE, labels)
    svm2.save(MODEL_PATH+'/'+'modelo_svm_cv2.dat')
    return svm2

def svm_classify3(features,labels):
    svm3 = svmLibsvm
    svm3.setKernel(cv2.cv2.ml.SVM_LINEAR)
    svm3.setType(cv2.cv2.ml.SVM_C_SVC)
    svm3.setC(2.67)
    svm3.setP(0.2)
#    svm.train(trainData,responses, params=svm_params)
    responses = np.array(labels).flatten()
    trainData = np.array(features)
#    svm2.train(trainData,responses, params=svm_params)
    svm3.train(trainData,cv2.cv2.ml.ROW_SAMPLE,responses)
#    svm2.train(features, cv2.ml.ROW_SAMPLE, labels)
    svm3.save(MODEL_PATH+'/'+'modelo_svm_cv2.dat')
    return svm3

@staticmethod
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


@staticmethod
def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), 
                          (0, 255, 0), thickness)


def readTransformMatrix(path, filename):
    with open(path+'/'+filename.lstrip('/'), 'r') as f:
        yaml.load(f)

def test_01():
    dataHogPositivos=hoggify(PATH_POSITIVAS,'jpg',False)
    dataHogNegativos=hoggify(PATH_NEGATIVAS,'jpg',False)
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
    
def test_02():
    extension='png'
    dataHogPositivos=hoggify(PATH_POSITIVAS_TRAIN,extension,False)
    dataHogNegativos=hoggify(PATH_NEGATIVAS_TRAIN,extension,False)
    n=len(dataHogPositivos)
    m=len(dataHogNegativos)
    dataHog=dataHogPositivos+dataHogNegativos
    labelsHog=[0]*n+[1]*m
    dataHogPositivosTest=hoggify(PATH_POSITIVAS_TEST,extension,False)
    dataHogNegativosTest=hoggify(PATH_NEGATIVAS_TEST,extension,False)
    nTest=len(dataHogPositivosTest)
    mTest=len(dataHogNegativosTest)
    dataHogTest=dataHogPositivosTest+dataHogNegativosTest
    labelsHogTest=[0]*nTest+[1]*mTest
    clf=svm_classify(dataHog,labelsHog)
    predicted=clf.predict(dataHogTest)
#    predicted_proba=clf.predict_proba(dataHogTest)
    print("Acurracy score: ",accuracy_score(labelsHogTest, predicted))
    
def test_03():
    dataHogPositivos=hoggify(PATH_POSITIVAS,'jpg',False)
    dataHogNegativos=hoggify(PATH_NEGATIVAS,'jpg',False)
    n=len(dataHogPositivos)
    m=len(dataHogNegativos)
    dataHog=dataHogPositivos+dataHogNegativos
    labelsHog=[0]*n+[1]*m
    svm2=svm_classify2(dataHog,labelsHog)
    X_test=dataHogPositivos[-1:]
    salida=svm2.predict(np.array(X_test))[1].ravel()
    print(salida)
#    probability=clf.predict_proba(X_test) # aqui calculo las probabilidades
#    log_probabilities=clf.predict_log_proba(X_test)
#    print("probabilidad 0: %f , 1:%f"%(math.exp(log_probabilities[0,0]),math.exp(log_probabilities[0,1])))
#    X_test=dataHogNegativos[-1:]
#    salida=clf.predict(X_test)
#    probability=clf.predict_proba(X_test) # aqui calculo las probabilidades
#    log_probabilities=clf.predict_log_proba(X_test)
#    print("probabilidad 0: %f , 1:%f"%(math.exp(log_probabilities[0,0]),math.exp(log_probabilities[0,1])))


#def detector(im,step_size,downscale):
#    im = imutils.resize(im, width = min(400, im.shape[1]))
#    min_wdw_sz = (64, 128)
##    step_size = (10, 10)
##    downscale = 1.25
#    clf = joblib.load(os.path.join(model_path, 'svm.model'))
#
#    #List to store the detections
#    detections = []
#    #The current scale of the image 
#    scale = 0
#
#    for im_scaled in pyramid_gaussian(im, downscale = downscale):
#        #The list contains detections at the current scale
#        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
#            break
#        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
#            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
#                continue
#            im_window = color.rgb2gray(im_window)
#            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
#
#            fd = fd.reshape(1, -1)
#            pred = clf.predict(fd)
#
#            if pred == 1:
#                
#                if clf.decision_function(fd) > 0.5:
#                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
#                    int(min_wdw_sz[0] * (downscale**scale)),
#                    int(min_wdw_sz[1] * (downscale**scale))))
#                 
#
#            
#        scale += 1
#
#    clone = im.copy()
#
#    for (x_tl, y_tl, _, w, h) in detections:
#        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
#
#    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
#    sc = [score[0] for (x, y, score, w, h) in detections]
#    print "sc: ", sc
#    sc = np.array(sc)
#    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
#    print "shape, ", pick.shape
#
#    for(xA, yA, xB, yB) in pick:
#        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
#    
#    plt.axis("off")
#    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
#    plt.title("Raw Detection before NMS")
#    plt.show()
#
#    plt.axis("off")
#    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
#    plt.title("Final Detections after applying NMS")
#    plt.show()
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
BShistory =  1000
#kernel = np.ones((5,5),np.uint8)
kernel = cv2.cv2.getStructuringElement(cv2.cv2.MORPH_RECT,(9,15))
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
# kernel2 = cv2.cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (2, 2))
#kernel2 = cv2.cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (4, 7))
#kernel2 = cv2.cv2.getStructuringElement(cv2.cv2.MORPH_RECT,(9,15))
min_thresh=50
max_thresh=500
fgbg = cv2.cv2.createBackgroundSubtractorMOG2(BShistory,100,False)
#fgbg = cv2.createBackgroundSubtractorMOG2()
bandera=True


def test_04(videoFilePath, videoFileName):
    """ Test 04
    The winStride  parameter is a 2-tuple that dictates the “step size” in both
    the x and y location of the sliding window.
    The padding parameter is a tuple which indicates the number of pixels in
    both the x and y direction in which the sliding window ROI is “padded”
    prior to HOG feature extraction.
    """
    global bandera
    winStride = (8,8)
#    padding=(32,32)
    padding = (8,8)
#    padding=(30,70)
    locations = ((0,0),)
    scale = 1.05
    iniFrame = 0
#    endFrame=200
    endFrame = 10000
    connectivity = 4
    currentSingleFrameNumber=0
    capture = cv2.cv2.VideoCapture(os.path.join(videoFilePath, videoFileName))
#    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#    fgbg = cv2.createBackgroundSubtractorMOG2()
#    fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG2()
    assert(capture.isOpened())
    capture.set(cv2.cv2.CAP_PROP_POS_FRAMES,currentSingleFrameNumber)
    contadorFrames = 0
    for currentSingleFrameNumber in range(iniFrame,int(endFrame)):
        success, origFrameImg = capture.read()
        if success:
            image_results=origFrameImg.copy()
            cv2.cv2.imwrite('frame'+str(currentSingleFrameNumber)+'.png',image_results)
            contadorFrames += 1
#            dim = 128
#            img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
#            img = hogcv2.compute(img)
#            img = np.squeeze(img)
#            hist = hogcv2.compute(image_results,winStride,padding,locations)
#            print(hist.shape)
#            print(hist[:36])
            fgmask = fgbg.apply(image_results)
#            fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
#            fgmask = cv2.erode(fgmask,kernel,iterations = 1)
            fgmask = cv2.cv2.morphologyEx(fgmask, cv2.cv2.MORPH_OPEN, kernel2)
            output = cv2.cv2.connectedComponentsWithStats(fgmask, connectivity, 
                                                      cv2.cv2.CV_32S)
            for i in range(output[0]):
                if output[2][i][4] >= min_thresh and output[2][i][4] <= max_thresh:
                    cv2.cv2.rectangle(image_results, 
                  (output[2][i][0], output[2][i][1]), (
                    output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)
#             dilation = cv2.dilate(erosion,kernel,iterations = 2)
#            fgmaskResized = cv2.resize(fgmask, (len(fgmask[0])//2,len(fgmask)//2), interpolation = cv2.INTER_AREA)
#            fgmaskResized = cv2.resize(fgmask, (len(fgmask[0])//2,len(fgmask)//2), interpolation = cv2.INTER_AREA)
            fgmaskResized = cv2.cv2.resize(image_results, (len(fgmask[0])//2,len(fgmask)//2), interpolation = cv2.cv2.INTER_AREA)
            fgmaskResized2 = cv2.cv2.resize(fgmask, (len(fgmask[0])//2,len(fgmask)//2), interpolation = cv2.cv2.INTER_AREA)
#            fgmaskResized = cv2.resize(dilation, (len(fgmask[0])//2,len(fgmask)//2), interpolation = cv2.INTER_AREA)
#            cv2.imshow('Results',image_results)
#            cv2.imshow('frame',fgmask)
            cv2.cv2.imshow('frame',fgmaskResized)
            cv2.cv2.imshow('frame2',fgmaskResized2)
        else:
            break
        key = cv2.cv2.waitKey(2)
#        key = cv2.waitKey(0)
        if key in [27, ord('Q'), ord('q')] or (not bandera):
            bandera=False
            break
    if capture is not None:
        capture.release()
    print(contadorFrames)
#    cv2.destroyAllWindows()
    
def test_05():
    global bandera
    bandera=True
    for i in range(100,250):
        if bandera:
            filename='chunk_'+str(i).zfill(6)+'.avi'
#            filename='chunk_'+str(i).zfill(6)+'.mp4'
            print(filename)
            test_04('/home/pepe/DATOS/Shared_Videos/TijuanaSantos/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks_TV',filename)
    cv2.cv2.destroyAllWindows()

def test_06():
    dataHogPositivosTest=hoggify(PATH_POSITIVAS_TEST,'jpg',False)
    dataHogNegativosTest=hoggify(PATH_NEGATIVAS_TEST,'jpg',False)
    nTest=len(dataHogPositivosTest)
    mTest=len(dataHogNegativosTest)
    dataHogTest=dataHogPositivosTest+dataHogNegativosTest
    labelsHogTest=[0]*nTest+[1]*mTest
    clf = joblib.load(MODEL_PATH+'/'+'modelo_svm_scikit.pkl', 'wb')
    predicted=clf.predict(dataHogTest)
#    predicted_proba=clf.predict_proba(dataHogTest)
    print("Acurracy score: ",accuracy_score(labelsHogTest, predicted))
    
def test_07():
    image_file_name='/home/pepe/DATOS/imagenes/MuestrasPlayers2/HumanosTest/GameConfig_107'
    extension='jpg'
    dataHogSample=hoggify_image(image_file_name,extension,False)
    clf = joblib.load(MODEL_PATH+'/'+'modelo_svm_scikit.pkl', 'wb')
    predicted=clf.predict_proba(dataHogSample)
    print(predicted[0,0])
    with open('./probability.dat', 'w') as f:
        f.write(str(predicted[0,0]))
        f.close()
    
def test_08():
    listOfFiles=[]
    dataHogToTest=hoggify_folder(PATH_POSITIVAS_TEST,'jpg',False,listOfFiles)
    nTest=len(dataHogToTest)
    clf = joblib.load(MODEL_PATH+'/'+'modelo_svm_scikit.pkl', 'wb')
#    predicted=clf.predict(dataHogToTest)
    predicted_proba=clf.predict_proba(dataHogToTest)
    with open('./probability.dat', 'w') as f:
        for i in range(nTest):
            print("%s %f"%(listOfFiles[i],predicted_proba[i,0]))
            f.write("%s %f\n"%(listOfFiles[i],predicted_proba[i,0]))
        f.close()
#    print("Acurracy score: ",accuracy_score(labelsHogTest, predicted))


def test_09():
    extesion = 'png'
    dataHogPositivosTest = hoggify(PATH_POSITIVAS_TEST, extesion, False)
    dataHogNegativosTest = hoggify(PATH_NEGATIVAS_TEST, extesion, False)
    nTest = len(dataHogPositivosTest)
    mTest = len(dataHogNegativosTest)
    dataHogTest = dataHogPositivosTest+dataHogNegativosTest
    labelsHogTest = [0]*nTest+[1]*mTest
    clf = joblib.load(MODEL_PATH+'/'+'modelo_svm_scikit.pkl', 'wb')
    predicted = clf.predict(dataHogTest)
    predicted_proba = clf.predict_proba(dataHogTest)
    print(predicted_proba)
    print(len(predicted_proba))
    print("Acurracy score: ", accuracy_score(labelsHogTest, predicted))



if __name__ == '__main__':
#    test_05()
    path = '/home/pepe/DATOS/Shared_Videos/TijuanaSantos'
    filename_game_config = 'GameConfig.Match'
    game_config_files = pputils.readGameConfigFile(path,filename_game_config)
    data_transform_matrix = pputils.reader_YAML(path,game_config_files[2])
    test_05()
    #testeando
#            print(line, end = '')
#    print(readGameConfigFile(path,filename))
    # with open(path+'/'+filename.lstrip('/'), 'r') as f:
    #     yaml.load(f)
    # fs = cv2.cv2.FileStorage(path+'/'+filename.lstrip('/'), cv2.cv2.FILE_STORAGE_READ)
    # fs.load()
    # # print(fs['MatchImageControlPoints'])
    # print(fs.getNode('MatchImageControlPoints'))
    # fs.release()


#    with cv2.cv2.FileStorage(path+'/'+filename.lstrip('/'), cv2.cv2.FILE_STORAGE_READ) as fs:
#        print(fs["MatchImageControlPoints"])  # Prints a matrix (read as a NumPy array)
    # fn = fs.getNode("MatchImageControlPoints")
    # print(np.asarray(fn))

#    test_09()
#    test_03()
#    responses=np.repeat(np.arange(2),250)[:,np.newaxis]
#    dataHogPositivos=hoggify(PATH_POSITIVAS,'jpg',False)
#    dataHogNegativos=hoggify(PATH_NEGATIVAS,'jpg',False)
#    n=len(dataHogPositivos)
#    m=len(dataHogNegativos)
#    dataHog=dataHogPositivos+dataHogNegativos
#    labelsHog=[0]*n+[1]*m
#    svm2=svm_classify2(dataHog,labelsHog)
#    X_test=dataHogPositivos[-1:]
#    salida=svm2.predict(np.array(X_test))[1].ravel()
#    salida=svm2.predict_proba(np.array(X_test))[1].ravel()
#    salida=svm2.predict(np.array(X_test),True)[1].ravel()
#    value=True
#    h1,h2=svm2.predict(np.array(X_test),value)
#    print(value)
#    pp = pprint.PrettyPrinter(indent=4)
#    pp.pprint(svm2)
#    pp.isreadable(svm2)
#    print(salida)