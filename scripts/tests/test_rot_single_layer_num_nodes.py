import sys
sys.path.append("../")

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from util.parser import Parser
from util.Evaluator import Evaluator
from util.Estimator import Estimator
from numpy import array

import json
import matplotlib.pyplot as plt
import keras.backend as K

def weighted_mean_squared_error(y_true, y_pred):
    difference = y_pred - y_true
    weights = array([20, 20, 20, 20, 1, 1, 1])
    return K.mean(K.square((difference*weights)), axis=-1)

outputFile = open(sys.argv[3],'w')
outputFile.write("Training data: " + sys.argv[1] + "\n")
outputFile.write("Testing data: " + sys.argv[2] + "\n")
outputFile.write("Trial Number,3 nodes rot_avg,3 nodes rot_max,"+ 
"4 nodes rot_avg,4 nodes rot_max,"+ 
"5 nodes rot_avg,5 nodes rot_max,"+ 
"6 nodes rot_avg,6 nodes rot_max,"+ 
"7 nodes rot_avg,7 nodes rot_max,"+ 
"8 nodes rot_avg,8 nodes rot_max,"+ 
"9 nodes rot_avg,9 nodes rot_max,"+ 
"10 nodes rot_avg,10 nodes rot_max,"+ 
"11 nodes rot_avg,11 nodes rot_max,"+ 
"12 nodes rot_avg,12 nodes rot_max,"+ 
"13 nodes rot_avg,13 nodes rot_max,"+ 
"14 nodes rot_avg,14 nodes rot_max,"+ 
"15 nodes rot_avg,15 nodes rot_max,"+ 
"16 nodes rot_avg,16 nodes rot_max,"+ 
"17 nodes rot_avg,17 nodes rot_max,"+ 
"18 nodes rot_avg,18 nodes rot_max,"+ 
"19 nodes rot_avg,19 nodes rot_max,"+ 
"20 nodes rot_avg,20 nodes rot_max,"+ 
"21 nodes rot_avg,21 nodes rot_max,"+ "\n")

for num_trial in range(1,4):
    print("Trial " + str(num_trial))
    outputFile.write(str(num_trial) + ",")
    for num_nodes in range(3,22,1):
        print(str(num_nodes) + " node layer...")
        model = Sequential()
        model.add(Dense(num_nodes, input_shape = (21,)))
        model.add(Dense(3))
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

        p = Parser()
        dataFileTrain = sys.argv[1]
        dataFileTest = sys.argv[2]

        inputDataTrain = array(p.Parse(dataFileTrain))
        print(inputDataTrain.shape)
        outputDataTrain = array(p.ParseSpineRotation(dataFileTrain))
        print(outputDataTrain.shape)
        history = model.fit(inputDataTrain, outputDataTrain, 32, 1000)

        inputDataTest = array(p.Parse(dataFileTest))
        print(inputDataTest.shape)
        outputDataTest = array(p.ParseSpineRotation(dataFileTest))
        print(outputDataTest.shape)
        loss_and_metrics = model.evaluate(inputDataTest, outputDataTest)
        print(loss_and_metrics)

        #test the model
        test = model.predict(inputDataTest, 1)

        #calculate differences
        e = Evaluator()
        result, avg, maxdiff = e.Difference(outputDataTest, test)

        print("max: " + str(maxdiff))
        print("avg: " + str(avg))
        
        avg_rot = sum(avg[:3])
        max_rot = sum(maxdiff[:3])

        print("sum avg_rot: " + str(avg_rot))
        print("sum max_rot: " + str(max_rot))
        outputFile.write(str(avg_rot) + ",")
        outputFile.write(str(max_rot) + ",")

    outputFile.write("\n")
outputFile.close()
