#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:40:55 2018

@author: pepe
"""

import numpy as np
import pandas as pd
import cv2

bandera=True

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

def get_folder_of_chunks(path):
    return path+'/folder_of_chunks'

def get_soccer_field_points(path,relative_path_transform_matrix):
    data_transform_matrix = reader_YAML(path,relative_path_transform_matrix)
    return np.array(data_transform_matrix[0][1]).reshape(-1, 2)

def read_lista_de_chunks(path):
    file = path+'/folder_of_chunks/lista_de_chunks.txt'
    numero_de_chunks = 0
    with open(file, 'r') as f:
        f.readline()#        No definido
        f.readline()#        /home/pepe/DATOS/PRUEBAS/TijuanaSantos/folder_of_chunks
        f.readline()#        chunk
        f.readline()#        200
        numero_de_chunks = int(f.readline())#        702
        total_de_frames = int(f.readline())#        24399
        terminador = f.readline()#        @
        f.close()
    return numero_de_chunks

def read_bounding_boxes(path,BoundigBox):
    data = pd.read_csv(path+BoundigBox)
    return data

def read_video_chunks(path,ini = 0,end = 0):
    global bandera
    bandera=True
    number_of_chunks = read_lista_de_chunks(path)
    if(ini < 0):
        print('Lo sentimos el primer indice debe ser mayor que 0!!!')
        return None
    if(number_of_chunks < end):
        print('Lo sentimos el número de chunks es mayor que los existentes {}'.format(number_of_chunks))
        return None
    for i in range(ini,end):
        if bandera:
            filename='chunk_'+str(i).zfill(6)+'.avi'
            print(path+'/'+filename)
#            test_04('/home/pepe/DATOS/Shared_Videos/TijuanaSantos/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks_TV',filename)
#    cv2.cv2.destroyAllWindows()

def read_game_id(path,game_id_file):
    file = path+game_id_file
    game_id = 0
    with open(file, 'r') as f:
        line = f.readline()
        while(line == '\n'):
            line = f.readline()
        game_id = int(line)
    return game_id

if __name__ == '__main__':
    path = '/home/pepe/DATOS/Shared_Videos/TijuanaSantos'
    filename_game_config = 'GameConfig.Match'
    game_config_files = readGameConfigFile(path,filename_game_config)
    soccer_field_points = get_soccer_field_points(path,game_config_files[2]) 
    number_of_chunks = read_lista_de_chunks(path)
    bounding_boxes = read_bounding_boxes(path,game_config_files[3])
    #read_video_chunks(path,0,10)
    game_id = read_game_id(path,game_config_files[1])
    print(game_id)