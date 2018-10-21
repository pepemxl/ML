#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:40:55 2018

@author: pepe
"""


def readGameConfigFile(path,filename):
    firstChunkFile = ''
    gameConfigFile = ''
    matrixROIFile = ''
    boundingBoxFile = ''
    with open(path+'/'+filename, 'r') as f:
        firstChunkFile = f.readline().rstrip('\n')
        gameConfigFile = f.readline().rstrip('\n')
        matrixROIFile = f.readline().rstrip('\n')
        boundingBoxFile = f.readline().rstrip('\n')
        f.close()
    return [firstChunkFile, gameConfigFile, matrixROIFile, boundingBoxFile]

def reader_YAML(path, filename, option = None):
    """
    Reader YAML 1.0
    @brief This reader only reads YAML that contains a str followed by a list
    of numbers.
    
    @todo create a set of options to read matrices or strings.
    """
    data=[]
    with open(path+'/'+filename.lstrip('/'), 'r') as f:
        f.readline()
        f.readline()
        data = f.read()
        data_lines = data.split(']')
        data_lines_splited = []
        for line in data_lines:
            if len(line) > 1:
                line = line.replace(' ', '')
                line_splited = line.split(':')
                line_splited[1] = line_splited[1].replace('[', '')
                line_splited[1] = line_splited[1].split(',')
                for i in range(len(line_splited[1])):
                    try:
                        float(line_splited[1][i])
                        line_splited[1][i] = float(line_splited[1][i])
                    except Exception:
                        print("Hubo un error al leer YAML ", end='')
                        print(line_splited[1][i])
                data_lines_splited.append(line_splited)
    return data_lines_splited

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
