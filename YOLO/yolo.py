# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import pputils
import pandas as pd
import csv
import cv2

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        #"score" : 0.01,
        "score" : 0.4,
        "iou" : 0.45,
        "model_image_size" : (608, 608),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, current_single_frame_number = 0, current_ROI = 0):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

#        print(image_data.shape) # la forma es el valor de configuracion!!!
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print(out_boxes)
        with open('/home/pp/Documents/CODIGOS/YOLO/keras/YOLO/path2your_video/box.dat','a',newline='\n') as csvfile:
            #csv_writer = csv.writer(csvfile,delimiter=',')
            csv_writer = csv.writer(csvfile)
            #csv_writer.writerow('#FRAME: {}'.format(current_single_frame_number))
            if current_ROI == 1:
                csvfile.write('#FRAME: {}'.format(current_single_frame_number))
                csvfile.write('\n')
#            for i in range(len(out_boxes)):
#                csv_writer.writerow(out_boxes[i])
#            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300
    
            for i, c in reversed(list(enumerate(out_classes))):
                row = []    
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
    
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
    
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                #print(label, (left, top), (right, bottom))
                row.append(score)
                row.append(left)
                row.append(top)
                row.append(right)
                row.append(bottom)
                csv_writer.writerow(row)
                pputils.game_id_array.append(pputils.game_id)
                pputils.frame_id_array.append(current_single_frame_number)
                punto = [int((left+right)//2),int(bottom)]
#                punto = pputils.convert_point_from_ROI(pputils.soccer_field_points, punto, current_ROI)
                punto = pputils.convert_point_from_ROIMxN(pputils.soccer_field_points, punto, current_ROI,pputils.M_rows,pputils.N_cols)
                pputils.x_original_array.append(punto[0])
                pputils.y_original_array.append(punto[1])
                pputils.x_field_array.append(0)# todo
                pputils.y_field_array.append(0)# todo
                pputils.width_array.append(abs(right-left))
                pputils.height_array.append(abs(bottom-top))
                pputils.probability_array.append(score)
#                print('{},{},{},{},{}'.format(score,left,top,right,bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
    
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        end = timer()
#        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def proccess_video_chunk(yolo, video_file_name, current_chunk):
    iniFrame = 0
    endFrame = 10000
    currentSingleFrameNumber=0
    capture = cv2.cv2.VideoCapture(video_file_name)
    contadorFrames = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    for currentSingleFrameNumber in range(iniFrame,int(endFrame)):
        success, origFrameImg = capture.read()
        if success:
            contadorFrames = currentSingleFrameNumber+current_chunk*200
            print(contadorFrames)
            image_results=origFrameImg.copy()
            image2 = cv2.cv2.bitwise_and(image_results,image_results,mask=pputils.mask)
#            [ROI1, ROI2, ROI3, ROI4] = pputils.create_ROIs(pputils.soccer_field_points,image2)
            ROI_array = pputils.create_ROIsMxN(pputils.soccer_field_points, image2,pputils.M_rows,pputils.N_cols)
            ROI_result = []
            for i in range(pputils.M_rows*pputils.N_cols):
                image = Image.fromarray(ROI_array[i])
                image = yolo.detect_image(image, contadorFrames,i+1)
                resulti = np.asarray(image)
                ROI_result.append(resulti)
#            [ROI1, ROI2, ROI3, ROI4, ROI5, ROI6, ROI7, ROI8] = pputils.create_ROIs8(pputils.soccer_field_points,image2)
#            image = Image.fromarray(ROI1)
#            image = yolo.detect_image(image, contadorFrames,1)
#            result1 = np.asarray(image)
#            image = Image.fromarray(ROI2)
#            image = yolo.detect_image(image, contadorFrames,2)
#            result2 = np.asarray(image)
#            image = Image.fromarray(ROI3)
#            image = yolo.detect_image(image, contadorFrames,3)
#            result3 = np.asarray(image)
#            image = Image.fromarray(ROI4)
#            image = yolo.detect_image(image, contadorFrames,4)
#            result4 = np.asarray(image)
#            image = Image.fromarray(ROI5)
#            image = yolo.detect_image(image, contadorFrames,5)
#            result5 = np.asarray(image)
#            image = Image.fromarray(ROI6)
#            image = yolo.detect_image(image, contadorFrames,6)
#            result6 = np.asarray(image)
#            image = Image.fromarray(ROI7)
#            image = yolo.detect_image(image, contadorFrames,7)
#            result7 = np.asarray(image)
#            image = Image.fromarray(ROI8)
#            image = yolo.detect_image(image, contadorFrames,8)
#            result8 = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
#            result = pputils.create_assemble_image_from_ROIs(pputils.soccer_field_points,image2, result1, result2, result3, result4)
#            result = pputils.create_assemble_image_from_ROIs8(pputils.soccer_field_points,image2, result1, result2, result3, result4, result5, result6, result7, result8)
            result = pputils.create_assemble_image_from_ROIsMxN(pputils.soccer_field_points,image2, ROI_result, pputils.M_rows, pputils.N_cols)
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2.50, color=(255, 255, 255), thickness=3)
            image_results = result
            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
            cv2.cv2.imshow('frame',image_resized)
#            [ROI1,ROI2,ROI3,ROI4] = pputils.create_ROIs(pputils.soccer_field_points, image2)
#            image_results = result1
#            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
#            cv2.cv2.imshow('frame1',image_resized)
#            image_results = result2
#            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
#            cv2.cv2.imshow('frame2',image_resized)
#            image_results = result3
#            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
#            cv2.cv2.imshow('frame3',image_resized)
#            image_results = result4
#            image_resized = cv2.cv2.resize(image_results, (len(image_results[0])//2,len(image_results)//2), interpolation = cv2.cv2.INTER_AREA)
#            cv2.cv2.imshow('frame4',image_resized)
#            contadorFrames += 1
        else:
            break
        key = cv2.cv2.waitKey(1)
        if key in [27, ord('Q'), ord('q')] or (not pputils.bandera):
            pputils.bandera = False
            break
    if capture is not None:
        capture.release()
        

def read_video_chunks(yolo, path,ini = 0,end = 0):
    pputils.bandera = True
    number_of_chunks = pputils.read_lista_de_chunks(path)
    if(ini < 0):
        print('Lo sentimos el primer indice debe ser mayor que 0!!!')
        return None
    if(number_of_chunks < end):
        print('Lo sentimos el número de chunks es mayor que los existentes {}'.format(number_of_chunks))
        return None
    for i in range(ini,end):
        if pputils.bandera:
            filename='chunk_'+str(i).zfill(6)+'.avi'
            print(path+'/folder_of_chunks/'+filename)
            pputils.initialize_arrays(pputils.game_id_array,
                   pputils.frame_id_array,
                   pputils.x_original_array,
                   pputils.y_original_array,
                   pputils.x_field_array,
                   pputils.y_field_array,
                   pputils.width_array,
                   pputils.height_array,
                   pputils.probability_array)
            proccess_video_chunk(yolo, path+'/folder_of_chunks/'+filename,i)
            query = pputils.insert_tblyolo('cobraj06','tblyolo',
                   pputils.game_id_array,
                   pputils.frame_id_array,
                   pputils.x_original_array,
                   pputils.y_original_array,
                   pputils.x_field_array,
                   pputils.y_field_array,
                   pputils.width_array,
                   pputils.height_array,
                   pputils.probability_array)
            pputils.execute_query(query)
            pputils.cnx.commit()
    cv2.cv2.destroyAllWindows()



def detect_video(yolo, video_path, output_path = ""):
#if __name__ == '__main__':
    video_path = '/home/pp/Documents/DATOS/GOLSTATS/PartidoTracking/TijuanaSantos'        
    path = video_path
    filename_game_config = 'GameConfig.Match'
    game_config_files = pputils.readGameConfigFile(path,filename_game_config)
    pputils.soccer_field_points = pputils.get_soccer_field_points(path,game_config_files[2]) 
    number_of_chunks = pputils.read_lista_de_chunks(path)
    pputils.bounding_boxes = pputils.read_bounding_boxes(path,game_config_files[3])
    pputils.game_id = pputils.read_game_id(path,game_config_files[1])
    pputils.create_mask(path)
#    pputils.read_video_chunks(path,0,100)
    pputils.connect_to_database()
    pputils.initialize_arrays(pputils.game_id_array, 
                              pputils.frame_id_array, 
                              pputils.x_original_array,
                              pputils.y_original_array, 
                              pputils.x_field_array, 
                              pputils.y_field_array,
                              pputils.width_array, 
                              pputils.height_array, 
                              pputils.probability_array)
    read_video_chunks(yolo,path,0,700)
    pputils.desconnect_from_database()
    
#    contador = 0
#    import cv2
#    vid = cv2.cv2.VideoCapture(video_path)
#    if not vid.isOpened():
#        raise IOError("Couldn't open webcam or video")
#    video_FourCC    = int(vid.get(cv2.cv2.CAP_PROP_FOURCC))
#    video_fps       = vid.get(cv2.cv2.CAP_PROP_FPS)
##    video_size      = (int(vid.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)),
##                        int(vid.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)))
#    video_size      = (int(x2-x1),int(y2-y1))
#    output_path = './path2your_video/salida.avi'
#    isOutput = True if output_path != "" else False
#    if isOutput:
#        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
#        fourcc = cv2.cv2.VideoWriter_fourcc('X','V','I','D')
#        out = cv2.cv2.VideoWriter(output_path, fourcc, video_fps, video_size)
#        output_path_original = './path2your_video/original.avi'
#        out_original = cv2.cv2.VideoWriter(output_path_original, fourcc, video_fps, video_size)
#        if not out.isOpened():
#            retval = out.open(output_path,video_FourCC,video_fps,video_size)
#            print('Algo salio muy mal con el codec')
#    accum_time = 0
#    curr_fps = 0
#    fps = "FPS: ??"
#    prev_time = timer()
#    bandera = True
#    vid.set(cv2.cv2.CAP_PROP_POS_FRAMES,120000)
#    while bandera:
#        return_value, frame = vid.read()
#        bandera = return_value
#        #print("bandera {}".format(bandera))
#        frame = frame[y1:y2, x1:x2]
#        image_original = frame.copy()
#        image = Image.fromarray(frame)
#        image = yolo.detect_image(image,contador)
#        result = np.asarray(image)
#        curr_time = timer()
#        exec_time = curr_time - prev_time
#        prev_time = curr_time
#        accum_time = accum_time + exec_time
#        curr_fps = curr_fps + 1
#        if accum_time > 1:
#            accum_time = accum_time - 1
#            fps = "FPS: " + str(curr_fps)
#            curr_fps = 0
#        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                    fontScale=0.50, color=(255, 0, 0), thickness=1)
#        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
##        resultResize = cv2.cv2.resize(result,(len(result[0])//2,len(result)//2),interpolation = cv2.cv2.INTER_AREA)
#        resultResize = result.copy()
#        cv2.imshow("result", resultResize)
#        if isOutput:
#            out.write(result)
#            out_original.write(image_original)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#        contador += 1
#    cv2.cv2.destroyAllWindows()
#    yolo.close_session()

