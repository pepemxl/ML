#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:40:55 2018

@author: pepe

"""

import numpy as np
import pandas as pd
import cv2
import mysql.connector


bandera = True
game_id = None
bounding_boxes = []
soccer_field_points = []
mask = []
cnx = None
game_id_array = []
frame_id_array = []
x_original_array = []
y_original_array = []
x_field_array = []
y_field_array = []
width_array = []
height_array = []
probability_array = []
M_rows = 2
N_cols = 8
_video_capture = None
_video_capture_file = None
vector_INI = []
vector_SIZE = []


def readGameConfigFile(path, filename):
    """
    @brief  Read Game Config File
    """
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
    data = []
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
    """
    @brief Convert a list in a matrix
    """
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
    theta[theta < 0] = np.pi + theta[theta < 0]
    return (s, theta)


def get_folder_of_chunks(path):
    return path+'/folder_of_chunks'


def get_soccer_field_points(path, relative_path_transform_matrix):
    data_transform_matrix = reader_YAML(path, relative_path_transform_matrix)
    return np.array(data_transform_matrix[0][1]).reshape(-1, 2)


def read_lista_de_chunks(path):
    file = path+'/folder_of_chunks/lista_de_chunks.txt'
    numero_de_chunks = 0
    with open(file, 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        numero_de_chunks = int(f.readline())
        total_de_frames = int(f.readline())
        terminador = f.readline()
        if terminador != '@\n':
            print('Un error al leer '+file)
            print(terminador)
        f.close()
    return numero_de_chunks


def read_bounding_boxes(path, BoundigBox):
    data = pd.read_csv(path+BoundigBox)
    return data


def sign(p1, p2, p3):
    signo = (p1[0]-p3[0])*(p2[1]-p3[1])-(p2[0]-p3[0])*(p1[1]-p3[1])
    return signo


def point_inside_triangle(p, v1, v2, v3):
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def point_in_field(i, j):
    global soccer_field_points
    if soccer_field_points.shape == (4, 2):
        p1 = soccer_field_points[0]
        p2 = soccer_field_points[1]
        p3 = soccer_field_points[2]
        p4 = soccer_field_points[3]
    else:
        print('Un error en point in field, incorrect shape!!!')
        return False
    if point_inside_triangle([i, j], p1, p2, p3) or point_inside_triangle([i, j], p1, p3, p4):
        return True
    else:
        return False


def create_ROIs(soccer_field_points_, image_):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    ROI1 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int(q1[0]):int((q1[0]+q2[0])//2)]
    ROI2 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//2):int(q2[0])]
    ROI3 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//2):int(q2[0])]
    ROI4 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int(q1[0]):int((q1[0]+q2[0])//2)]
    return [ROI1, ROI2, ROI3, ROI4]


def create_ROIs8(soccer_field_points_, image_):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    ROI1 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int(q1[0]):int((q1[0]+q2[0])//4)]
    ROI2 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//4):int((q1[0]+q2[0])//2)]
    ROI3 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//2):int(3*((q1[0]+q2[0])//4))]
    ROI4 = image_[int(q1[1]):int((q1[1]+q2[1])//2), int(3*((q1[0]+q2[0])//4)):int(q2[0])]
    ROI5 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int(3*((q1[0]+q2[0])//4)):int(q2[0])]
    ROI6 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//2):int(3*((q1[0]+q2[0])//4))]
    ROI7 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//4):int((q1[0]+q2[0])//2)]
    ROI8 = image_[int((q1[1]+q2[1])//2):int(q2[1]), int(q1[0]):int((q1[0]+q2[0])//4)]
    return [ROI1, ROI2, ROI3, ROI4, ROI5, ROI6, ROI7, ROI8]

def create_ROIsMxN(soccer_field_points_, image_, M, N):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    ROI_array = []
    for i in range(M):
        for j in range(N):
            ROI = image_[int(i*((q2[1]-q1[1])//M)+q1[1]):int((i+1)*((q2[1]-q1[1])//M)+q1[1]), int(j*((q2[0]-q1[0])//N)+q1[0]):int((j+1)*((q2[0]-q1[0])//N)+q1[0])]
            ROI_array.append(ROI)
    return ROI_array


def create_assemble_image_from_ROIs(soccer_field_points_, image_, ROI1, ROI2, ROI3, ROI4):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    image = image_.copy()
    image[int(q1[1]):int((q1[1]+q2[1])//2), int(q1[0]):int((q1[0]+q2[0])//2)] = ROI1
    image[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//2):int(q2[0])] = ROI2
    image[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//2):int(q2[0])] = ROI3
    image[int((q1[1]+q2[1])//2):int(q2[1]), int(q1[0]):int((q1[0]+q2[0])//2)] = ROI4
    return image


def create_assemble_image_from_ROIs8(soccer_field_points_, image_, ROI1, ROI2, ROI3, ROI4, ROI5, ROI6, ROI7, ROI8):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    image = image_.copy()
    image[int(q1[1]):int((q1[1]+q2[1])//2), int(q1[0]):int((q1[0]+q2[0])//4)] = ROI1
    image[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//4):int((q1[0]+q2[0])//2)] = ROI2
    image[int(q1[1]):int((q1[1]+q2[1])//2), int((q1[0]+q2[0])//2):int(3*((q1[0]+q2[0])//4))] = ROI3
    image[int(q1[1]):int((q1[1]+q2[1])//2), int(3*((q1[0]+q2[0])//4)):int(q2[0])] = ROI4
    image[int((q1[1]+q2[1])//2):int(q2[1]), int(3*((q1[0]+q2[0])//4)):int(q2[0])] = ROI5
    image[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//2):int(3*((q1[0]+q2[0])//4))] = ROI6
    image[int((q1[1]+q2[1])//2):int(q2[1]), int((q1[0]+q2[0])//4):int((q1[0]+q2[0])//2)] = ROI7
    image[int((q1[1]+q2[1])//2):int(q2[1]), int(q1[0]):int((q1[0]+q2[0])//4)] = ROI8
    return image


def create_assemble_image_from_ROIsMxN(soccer_field_points_, image_,ROI_array, M, N):
    soccer_field_points_ = soccer_field_points
    q1 = [min(soccer_field_points_[:, 0]), min(soccer_field_points_[:, 1])]
    q2 = [max(soccer_field_points_[:, 0]), max(soccer_field_points_[:, 1])]
    image = image_.copy()
    for i in range(M):
        for j in range(N):
            image[int(i*((q2[1]-q1[1])//M)+q1[1]):int((i+1)*((q2[1]-q1[1])//M)+q1[1]), int(j*((q2[0]-q1[0])//N)+q1[0]):int((j+1)*((q2[0]-q1[0])//N)+q1[0])] = ROI_array[N*i+j]
    return image


def convert_point_from_ROI(soccer_field_points_, p, n_ROI):
    q1 = [int(min(soccer_field_points_[:, 0])), int(min(soccer_field_points_[:, 1]))]
    q2 = [int(max(soccer_field_points_[:, 0])), int(max(soccer_field_points_[:, 1]))]
    if n_ROI == 1:
        q = p+q1
    if n_ROI == 2:
        q = p+[int((q1[0]+q2[0])//2), int(q1[1])]
    if n_ROI == 3:
        q = p+[int((q1[0]+q2[0])//2), int((q1[1]+q2[1])//2)]
    if n_ROI == 4:
        q = p+[int(q1[0]), int((q1[1]+q2[1])//2)]
    return q


def convert_point_from_ROI8(soccer_field_points_, p, n_ROI):
    q1 = [int(min(soccer_field_points_[:, 0])), int(min(soccer_field_points_[:, 1]))]
    q2 = [int(max(soccer_field_points_[:, 0])), int(max(soccer_field_points_[:, 1]))]
    if n_ROI == 1:
        q = p+q1
    if n_ROI == 2:
        q = p+[int((q1[0]+q2[0])//4), int(q1[1])]
    if n_ROI == 3:
        q = p+[int((q1[0]+q2[0])//2), int(q1[1])]
    if n_ROI == 4:
        q = p+[int(3*((q1[0]+q2[0])//4)), int(q1[1])]
    if n_ROI == 5:
        q = p+[int(3*((q1[0]+q2[0])//4)), int((q1[1]+q2[1])//2)]
    if n_ROI == 6:
        q = p+[int((q1[0]+q2[0])//2), int((q1[1]+q2[1])//2)]
    if n_ROI == 7:
        q = p+[int((q1[0]+q2[0])//4), int((q1[1]+q2[1])//2)]
    if n_ROI == 8:
        q = p+[int(q1[0]), int((q1[1]+q2[1])//2)]
    return q


def convert_point_from_ROIMxN(soccer_field_points_, p, n_ROI, M, N):
    q1 = [int(min(soccer_field_points_[:, 0])), int(min(soccer_field_points_[:, 1]))]
    q2 = [int(max(soccer_field_points_[:, 0])), int(max(soccer_field_points_[:, 1]))]
    n_row = n_ROI//N
    n_col = n_ROI%N
    q = p + [int(n_col*((q2[0]-q1[0])//N)+q1[0]), int(n_row*((q2[1]-q1[1])//M)+q1[1])]
    return q


def create_mask(path):
    global soccer_field_points
    global mask
    filename = 'chunk_' + str(0).zfill(6) + '.avi'
    video_file_name = path+'/folder_of_chunks/'+filename
    capture = cv2.cv2.VideoCapture(video_file_name)
    success, im_mask = capture.read()
    if success:
        height, width, depth = im_mask.shape
        mask = np.zeros((height, width), np.uint8)
# Anterior method, very slow
#        for i in range(height):
#            for j in range(width):
#                if point_in_field(j,i):
#                    mask[i,j] = 255
# Creando version optimizada
        pts = np.squeeze(np.int32(soccer_field_points))
#        print(pts)
#        cv2.cv2.polylines(mask, [pts], True, (255,255,255), 3)
#        cv2.cv2.fillPoly(img, pts =[contours], color=(255,255,255))
        cv2.cv2.fillPoly(mask, [pts], color=(255, 255, 255))
#        cv2.circle(mask,(100,100), 50, (255,255,255), -1)
#        cv2.circle(mask, tuple(arreglo), 50, (255,255,255), -1)
#        cv2.cv2.rectangle(mask,(x2,y2),(x2+w2,y2+h2),(255,255,255),5)
    else:
        print('Hubo un error al intentar crear la mascara')
#    image_resized = cv2.cv2.resize(mask, (len(mask[0])//2,len(mask)//2), interpolation = cv2.cv2.INTER_AREA)
#    cv2.cv2.imshow('mascara',image_resized)
#    cv2.cv2.waitKey(0)
#    cv2.cv2.destroyAllWindows()


def read_frame_from_chunk(path, current_chunk_number, current_frame_on_chunk):
    global bandera
    global mask
    global soccer_field_points
    global M_rows
    global N_cols
    global _video_capture
    global _video_capture_file
    filename = 'chunk_'+str(current_chunk_number).zfill(6)+'.avi'
    video_file_name = path+'/folder_of_chunks/'+filename
    if video_file_name != _video_capture_file:
        try:
            _video_capture.release()
        except:
            print('Error al intentar liberar capturadora')
        _video_capture = cv2.cv2.VideoCapture(video_file_name)
        _video_capture_file = video_file_name
    if(not _video_capture.isOpened()):
        _video_capture = cv2.cv2.VideoCapture(video_file_name)
    current_pos = _video_capture.get(cv2.cv2.CAP_PROP_POS_FRAMES)
    if current_pos != current_frame_on_chunk:
        print("Cambio {} -> {}".format(current_pos, current_frame_on_chunk))
        _video_capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, current_frame_on_chunk)
    success, origFrameImg = _video_capture.read()
    if success:
        image2 = cv2.cv2.bitwise_and(origFrameImg, origFrameImg, mask=mask)
        return image2
    return None


def proccess_video_chunk(video_file_name):
    global bandera
    global mask
    global soccer_field_points
    global M_rows
    global N_cols
    iniFrame = 0
#    endFrame=200
    endFrame = 10000
    currentSingleFrameNumber=0
    capture = cv2.cv2.VideoCapture(video_file_name)
    contadorFrames = 0
#    print(soccer_field_points)
    for currentSingleFrameNumber in range(iniFrame,int(endFrame)):
        success, origFrameImg = capture.read()
        if success:
            image_results=origFrameImg.copy()
            image2 = cv2.cv2.bitwise_and(image_results,image_results,mask=mask)
#            [ROI1,ROI2,ROI3,ROI4] = create_ROIs(soccer_field_points, image2)
#            [ROI1,ROI2,ROI3,ROI4,ROI5,ROI6,ROI7,ROI8] = create_ROIs8(soccer_field_points, image2)
            ROI_array = create_ROIsMxN(soccer_field_points, image2,M_rows,N_cols)
#            for i in range(M_rows*N_cols):
#                image_results = ROI_array[i]
#                image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
#                windows_name = 'frame '+str(i)
#                cv2.cv2.imshow(windows_name,image_resized)
            image3 = image2.copy()
            image3 = create_assemble_image_from_ROIsMxN(soccer_field_points,image3,ROI_array, M_rows, N_cols)
            image_results = image3
            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
            cv2.cv2.imshow('frame',image_resized)
#            cv2.cv2.imshow('mask',mask)
#            cv2.cv2.imwrite('frame'+str(currentSingleFrameNumber)+'.png',image_results)
            contadorFrames += 1
        else:
            break
        key = cv2.cv2.waitKey(1)
#        key = cv2.waitKey(0)
        if key in [27, ord('Q'), ord('q')] or (not bandera):
            bandera = False
            break
    if capture is not None:
        capture.release()


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
            print(path+'/folder_of_chunks/'+filename)
            proccess_video_chunk(path+'/folder_of_chunks/'+filename)
#            test_04('/home/pepe/DATOS/Shared_Videos/TijuanaSantos/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks',filename)
#            test_04('/home/pepe/DATOS/LigaMX/Jornada_07/chunksPrueba30/folder_of_chunks_TV',filename)
    cv2.cv2.destroyAllWindows()


def read_game_id(path, game_id_file):
    file = path+game_id_file
    game_id = 0
    with open(file, 'r') as f:
        line = f.readline()
        while(line == '\n'):
            line = f.readline()
        game_id = int(line)
    return game_id


def connect_to_database():
    global cnx
    cnx = mysql.connector.connect(user='pepe', password='pepemxl',
                                  host='172.16.0.108',
                                  database='cobraj06')


def desconnect_from_database():
    global cnx
    cnx.close()


def execute_query(query):
    cursor = cnx.cursor()
    cursor.execute(query)
#    for (game_id, player_id, tracker_id) in cursor:
#        print("{}".format(game_id))
    return cursor


#function to test
def initialize_arrays(game_id_array, frame_id_array, x_original_array,
                      y_original_array, x_field_array, y_field_array,
                      width_array, height_array, probability_array):
    del game_id_array[:]
    del frame_id_array[:]
    del x_original_array[:]
    del y_original_array[:]
    del x_field_array[:]
    del y_field_array[:]
    del width_array[:]
    del height_array[:]
    del probability_array[:]


def insert_tblyolo(database_, table_,
                   game_id_, frame_id_, x_original_, y_original_,
                   x_field_, y_field_, width_, height_, probability_):
    query = 'INSERT INTO ' + database_ + '.' + table_ + ' '
    query += '(game_id,frame_id,x_original,y_original,x_field,y_field,width,height,probability) VALUES '
    for i in range(len(game_id_)):
        if i > 0:
            query += ','
        query += '('+str(game_id_[i]) + ',' + \
                 str(frame_id_[i]) + ',' + \
                 str(x_original_[i]) + ',' + \
                 str(y_original_[i]) + ',' + \
                 str(x_field_[i]) + ',' + \
                 str(y_field_[i]) + ',' + \
                 str(width_[i]) + ',' + \
                 str(height_[i]) + ',' + \
                 str(probability_[i]) + ')'
    return query


def select_yolo(database_, table_, game_id_, frame_id_ini_, frame_id_end_):
    query = 'SELECT game_id,frame_id,x_original,y_original,\
             x_field,y_field,width,height,probability FROM '\
             + database_ + '.' + table_ + ' '
    query += 'WHERE game_id=' + str(game_id_) + ' AND ' + \
             'frame_id between ' + str(frame_id_ini_) + \
             ' AND ' + str(frame_id_end_) + ' ORDER BY frame_id,probability'
    return query


def compute_vectors_INI_SIZE():
    global frame_id_array
    global vector_INI
    global vector_SIZE
    del vector_INI[:]
    del vector_SIZE[:]
    N_min = min(frame_id_array)
    N_max = max(frame_id_array)+1
    total = N_max - N_min
    N_registers = len(frame_id_array)
    vector_INI = [0]*total
    vector_SIZE = [0]*total
    for i in range(N_registers):
        if frame_id_array[i] < N_max + 1:
            vector_SIZE[frame_id_array[i]-N_min] += 1
    for i in range(1, total):
        vector_INI[i] = vector_SIZE[i-1] + vector_INI[i-1];
        
        
def draw_boxes_on_frame(image, current_single_frame_number):
    global vector_INI
    global vector_SIZE
    global frame_id_array
    global x_original_array
    global y_original_array
    global x_field_array
    global y_field_array
    global width_array
    global height_array
    global probability_array
    N_min = min(frame_id_array)
    total = vector_SIZE[current_single_frame_number-N_min]
    for i in range(total):
        print('hola mundo')



def test_01_arrays(game_id_array, frame_id_array, x_original_array,
                   y_original_array, x_field_array, y_field_array,
                   width_array, height_array, probability_array):
    game_id_array.append(500)
    frame_id_array.append(200)
    x_original_array.append(20.678)
    y_original_array.append(12.879123)
    x_field_array.append(13.789243)
    y_field_array.append(14.7684)
    width_array.append(300)
    height_array.append(400)
    probability_array.append(.98972)


def test_02_arrays(game_id_array, frame_id_array, x_original_array,
                   y_original_array, x_field_array, y_field_array,
                   width_array, height_array, probability_array):
    game_id_array.append(500)
    frame_id_array.append(100)
    x_original_array.append(10.678)
    y_original_array.append(2.879123)
    x_field_array.append(3.789243)
    y_field_array.append(4.7684)
    width_array.append(30)
    height_array.append(40)
    probability_array.append(.8972)

def test_03_bounding_boxes_from_database(path, frame_ini, frame_end):
    global bandera
    bandera = True
    number_of_chunks = read_lista_de_chunks(path)
    initialize_arrays(game_id_array, frame_id_array, x_original_array,
                      y_original_array, x_field_array, y_field_array,
                      width_array, height_array, probability_array)
    connect_to_database()
    query = select_yolo('cobraj06', 'tblyolo', 50106, frame_ini, frame_end)
    cursor = execute_query(query)
    for (game_id, frame_id, x_original, y_original, x_field, y_field, width, height, probability) in cursor:
        game_id_array.append(game_id)
        frame_id_array.append(frame_id)
        x_original_array.append(x_original)
        y_original_array.append(y_original)
        x_field_array.append(x_field)
        y_field_array.append(y_field)
        width_array.append(width)
        height_array.append(height)
        probability_array.append(probability)
    compute_vectors_INI_SIZE()
    key = cv2.cv2.waitKey(1)
    for current_frame_counter in range(frame_ini, frame_end+1):
#        print('{} {} {} {} {} {} {} {} {}'.format(game_id,frame_id,x_original,y_original,x_field,y_field,width,height,probability))
        current_chunk_number = current_frame_counter//200
        curren_frame_in_chunk = current_frame_counter % 200
        current_frame = read_frame_from_chunk(path, current_chunk_number, curren_frame_in_chunk)
        if len(current_frame) > 0:
            draw_boxes_on_frame(current_frame, current_frame_counter)
            image_resized = cv2.cv2.resize(current_frame, (len(current_frame[0])//2, len(current_frame)//2), interpolation = cv2.cv2.INTER_AREA)
            cv2.cv2.imshow('Test', image_resized)
            key = cv2.cv2.waitKey(1)
        if key in [27, ord('Q'), ord('q')] or (not bandera):
            bandera = False
            break
    cv2.cv2.destroyAllWindows()
    desconnect_from_database()


if __name__ == '__main__':
    path = '/home/pepe/DATOS/Shared_Videos/TijuanaSantos'
    filename_game_config = 'GameConfig.Match'
    game_config_files = readGameConfigFile(path, filename_game_config)
    soccer_field_points = get_soccer_field_points(path, game_config_files[2])
    number_of_chunks = read_lista_de_chunks(path)
    bounding_boxes = read_bounding_boxes(path, game_config_files[3])
    game_id = read_game_id(path, game_config_files[1])
    create_mask(path)
    test_03_bounding_boxes_from_database(path, 200*1, 200*2-1)
#    print(cv2.CAP_PROP_POS_FRAMES)
    #    read_video_chunks(path, 0, 100)
#    cursor.close()
#    test_01_arrays(game_id_array, frame_id_array, x_original_array,
#                      y_original_array, x_field_array, y_field_array,
#                      width_array, height_array, probability_array)
#    test_02_arrays(game_id_array, frame_id_array, x_original_array,
#                      y_original_array, x_field_array, y_field_array,
#                      width_array, height_array, probability_array)
#    query = insert_tblyolo('cobraj06','tblyolo',game_id_array, frame_id_array, x_original_array,
#                      y_original_array, x_field_array, y_field_array,
#                      width_array, height_array, probability_array)
    
#    execute_query(query)
#    cnx.commit()
##    cnx.rollback()


#    print(bounding_boxes)