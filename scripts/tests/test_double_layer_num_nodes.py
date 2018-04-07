import sys
sys.path.append("../")

from keras.models import Model
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
outputFile.write("Trial Number,5 nodes rot_avg,5 nodes rot_max,5 nodes pos_avg,5 nodes pos_max,"+ 
"10 nodes rot_avg,10 nodes rot_max,10 nodes pos_avg,10 nodes pos_max,"+ 
"15 nodes rot_avg,15 nodes rot_max,15 nodes pos_avg,15 nodes pos_max,"+ 
"20 nodes rot_avg,20 nodes rot_max,20 nodes pos_avg,20 nodes pos_max,"+ 
"25 nodes rot_avg,25 nodes rot_max,25 nodes pos_avg,25 nodes pos_max,"+ 
"30 nodes rot_avg,30 nodes rot_max,30 nodes pos_avg,30 nodes pos_max,"+
"35 nodes rot_avg,35 nodes rot_max,35 nodes pos_avg,35 nodes pos_max,"+
"40 nodes rot_avg,40 nodes rot_max,40 nodes pos_avg,40 nodes pos_max,"+
"45 nodes rot_avg,45 nodes rot_max,45 nodes pos_avg,45 nodes pos_max,"+
"50 nodes rot_avg,50 nodes rot_max,50 nodes pos_avg,50 nodes pos_max,"+"\n")

for num_trial in range(1,4):
    print("Trial " + str(num_trial))
    outputFile.write(str(num_trial) + ",")
    for num_nodes in range(5,55,5):
        print(str(num_nodes) + " node layer...")
        inputLayer = Input(shape=(21,))
        hiddenLayer1 = Dense(num_nodes)(inputLayer)
        hiddenLayer2 = Dense(11)(hiddenLayer1)
        outputLayer = Dense(7)(hiddenLayer2)
        model = Model(inputs=inputLayer, outputs=outputLayer)
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        p = Parser()
        dataFileTrain = sys.argv[1]
        dataFileTest = sys.argv[2]

        inputDataTrain = array(p.Parse(dataFileTrain))
        print(inputDataTrain.shape)
        outputDataTrain = array(p.ParseSpine(dataFileTrain))
        print(outputDataTrain.shape)
        history = model.fit(inputDataTrain, outputDataTrain, 32, 2000)

        inputDataTest = array(p.Parse(dataFileTest))
        print(inputDataTest.shape)
        outputDataTest = array(p.ParseSpine(dataFileTest))
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
        avg_pos = sum(avg[4:])

        max_rot = sum(maxdiff[:3])
        max_pos = sum(maxdiff[4:])

        print("sum avg_rot: " + str(avg_rot))
        print("sum avg_pos: " + str(avg_pos))
        print("sum max_rot: " + str(max_rot))
        print("sum max_pos: " + str(max_pos))
        outputFile.write(str(avg_rot) + ",")
        outputFile.write(str(max_rot) + ",")
        outputFile.write(str(avg_pos) + ",")
        outputFile.write(str(max_pos) + ",")

    outputFile.write("\n")
outputFile.close()
