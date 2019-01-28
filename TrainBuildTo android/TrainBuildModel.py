# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 23:34:17 2017

@author: 12102083
"""

# In[ ]: 
    #import cv2
    import os
    import numpy 
    import pandas 
    #from sklearn.utils import shuffle  
    from keras.wrappers.scikit_learn import KerasRegressor  
    #from sklearn.model_selection import cross_val_score 
    #from sklearn.model_selection import KFold
    #from sklearn.preprocessing import StandardScaler 
    #import matplotlib.pyplot as plt 
    import time 
    #from sklearn.pipeline import Pipeline 
    from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib 
    from keras.models import Sequential  
    from keras.layers import Dense, Activation 
    from keras.layers import Convolution2D, MaxPooling2D, Flatten
    from keras.optimizers import SGD 
    from keras.models import model_from_json   
    numpy.random.seed(seed=7)  
    import tensorflow as tf
    sess = tf.Session() 
    import keras.backend as K   
    
    K.set_session(sess)
    
# In[ ]:

# In[ ]:   
print("Loading Data")
X, y = # X Training data y = test
print("Data Loaded")
# In[]

#neural net 

model = Sequential() 
model.add(Convolution2D(32, 3, 3, input_shape=(96, 96,1))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Convolution2D(32, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) # this is the removed layer

model.add(Convolution2D(64, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Flatten()) 
model.add(Dense(1000)) 
model.add(Activation('relu')) 
model.add(Dense(500)) 
model.add(Activation('relu')) 
model.add(Dense(136)) 

print("Compling Model")
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy','mae'])   
print("Model Compiled")   

print("Fitting Model") 
tf.train.write_graph(sess.graph_def, '.', r'TFAndroid\mobile.pbtxt')
hist = model.fit(X, y, nb_epoch=1, batch_size=120, validation_split=0.1, verbose = 1) 
json_string = model.to_json() 
open(r'TFAndroid\Base.json','w').write(json_string) 
model.save_weights(r'TFAndroid\Base.h5')    

print("saving Data") 
numpy.savetxt(r"TFAndroid\loss_hist.txt", numpy.array(hist.history['loss']), delimiter=",") 
numpy.savetxt(r"TFAndroid\acc_hist.txt", numpy.array(hist.history['acc']), delimiter=",")
numpy.savetxt(r"TFAndroid\mean_absolute_error_hist.txt", numpy.array(hist.history['mean_absolute_error']), delimiter=",")
print("data saved")


# In[]   

print("RESET KERNAL BEFORE THIS HAPPENS")
import os
import numpy 
import pandas 
from keras.wrappers.scikit_learn import KerasRegressor  
import time 
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib 
from keras.models import Sequential  
from keras.layers import Dense, Activation 
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD 
from keras.models import model_from_json   
numpy.random.seed(seed=7)  
import tensorflow as tf
sess = tf.Session() 
import keras.backend as K   

K.set_session(sess)
with tf.Session() as sess:
    model = Sequential() 
    model.add(Convolution2D(32, 3, 3, input_shape=(96, 96,1))) 
    model.add(Activation('relu')) 
    
    model.add(Convolution2D(32, 2, 2)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2,2))) # this is the removed layer
    
    model.add(Convolution2D(64, 2, 2)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2,2))) 
    
    model.add(Flatten()) 
    model.add(Dense(1000)) 
    model.add(Activation('relu')) 
    model.add(Dense(500)) 
    model.add(Activation('relu')) 
    model.add(Dense(136))
    model.load_weights(r'TFAndroid\Base.h5')

    graph_def = sess.graph.as_graph_def() 
    saver = tf.train.Saver().save(sess, os.path.join(os.getcwd(),r'TFAndroid\model.ckpt'))
    tf.train.write_graph(graph_def, logdir='.',   name=r'TFAndroid\model.pb', as_text=False) 
    print("use this to try and get the final node : ")#[print(n.name) for n in graph_def.node]
    [print(n.name) for n in graph_def.node]
    MODEL_NAME = r'TFAndroid\model'
    
    # Freeze the graph
    
    print("Freezing the graph")
    OutputNodeName = 'add_5'
    input_graph_path = r'TFAndroid\mobile'+'.pbtxt'
    checkpoint_path = './'+MODEL_NAME+'.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = OutputNodeName
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = MODEL_NAME+'_frozen'+'.pb'
    output_optimized_graph_name = MODEL_NAME+'_optimized'+'.pb'
    clear_devices = True
    
    
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")  
    print("Finished freeze graph saved") 
    
    print("Optimizing the graph")
    input_graph_def = tf.GraphDef() 
    with tf.gfile.Open(output_frozen_graph_name, "r") as f: 
        data = f.read() 
        input_graph_def.ParseFromString(data) 
        
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            ['convolution2d_input_1'],
            [OutputNodeName],
            tf.float32.as_datatype_enum) 
    print("Graph Optimized") 
    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString()) 
    
    print("Optimized graph saved") 
    

 
# In[ ]: 
# predict the accuracy of the models  
import cv2
from os import listdir

model = LoadModelForRetrainingName(r'PathToModel')
cascPath = 'PathToOpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
path = r'PathToTestingData'
files = listdir(path) 
for file in files: 
    if "pts" not in file:
        print file 
        grey = cv2.imread(path+ '\\' + file,0)  
        face = cv2.imread(path+ '\\' + file)
        faces = faceCascade.detectMultiScale(grey,1.3, 5) 
        print(len(faces)) 
        if len(faces) == 1:
            for (x,y,w,h) in faces: 
                cropface = grey[y:y+h,x:x+w] 
                cropface = cv2.resize(cropface,(96,96))           
                cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0),2)  
                croptemp = cropface.reshape(-1,1,96,96) 
                divpic = croptemp / 255.0
                facePoints = model.predict(divpic)  
                x_points = facePoints[0][0::2]#*48+48 
                y_points = facePoints[0][1::2]#*48+48 
                for i in range(63):
                    #cv2.circle(face,(x_points[i],y_points[i]),3,(0,225,0),-1)    
                    cv2.circle(face,(int(x+(w*(x_points[i]/96))),int(y+(h*(y_points[i]/96)))),3,(255,255,255),-1)
            cv2.imshow("Rect",face) 
            
            cv2.imwrite('C:\\Users\\12102083\Desktop\\IMAGES\\Temp\\'+file,face)
        else: 
            print("Skipped Image : " + file + " As " + str(len(faces)) + " faces was detected")  
            cv2.imshow("Rect",grey)
        cv2.imshow("Temp",grey)  
        cv2.waitKey(1000) 

# In[ ]: 
import cv2
import sys
import os
import numpy
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle  
import pandas

from keras.models import Sequential  
from keras.layers import Dense, Activation 
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD 
from keras.models import model_from_json    

model = Sequential() 
model.add(Convolution2D(32, 3, 3, input_shape=(96, 96,1))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Convolution2D(32, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) # this is the removed layer

model.add(Convolution2D(64, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Flatten()) 
model.add(Dense(1000)) 
model.add(Activation('relu')) 
model.add(Dense(500)) 
model.add(Activation('relu')) 
model.add(Dense(136))
model.load_weights(r'TFAndroid\Base.h5') 
cascPath = 'PathToOpencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
printOneFrame = False 

empty = numpy.zeros(shape=(0,96,96,1))  
model.predict(empty)
print('Network Loaded!')
 

cap = cv2.VideoCapture(0) 
while(cap.isOpened()): 
    # get a frame
    ret, frame = cap.read()  
    cap.set(3,640) 
    cap.set(4,480)
    if(cap.read()): 
        grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) 
        faces = faceCascade.detectMultiScale(grey,1.3, 5) 
        for (x,y,w,h) in faces: 
            cropface = grey[y:y+h,x:x+w] 
            cropface = cv2.resize(cropface,(96,96))           
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)  
            croptemp = cropface.reshape(-1,96,96,1) 
            divpic = croptemp / 255.0
            facePoints = model.predict(divpic)  

            x_points = facePoints[0][0::2]#*48+48 
            y_points = facePoints[0][1::2]#*48+48 

            for i in range(64):
                cv2.circle(frame,(int(x+(w*(x_points[i]/96))),int(y+(h*(y_points[i]/96)))),3,(255,255,255),-1)

                print("")
        cv2.imshow('Webcam',frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
            break
cap.release()
