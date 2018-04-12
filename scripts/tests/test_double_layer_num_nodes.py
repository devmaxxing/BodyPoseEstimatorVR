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

for num_trial in range(1,3):
    print("Trial " + str(num_trial))
    outputFile = open(sys.argv[3] + "_" + str(num_trial),'w')
    outputFile.write("Training data: " + sys.argv[1] + "\n")
    outputFile.write("Testing data: " + sys.argv[2] + "\n")
    outputFile.write("Num nodes 2, Num nodes 1,5 nodes rot_avg,5 nodes rot_max,"+ 
    "10 nodes rot_avg,10 nodes rot_max,"+ 
    "15 nodes rot_avg,15 nodes rot_max,"+ 
    "20 nodes rot_avg,20 nodes rot_max,"+ 
    "25 nodes rot_avg,25 nodes rot_max,"+ 
    "30 nodes rot_avg,30 nodes rot_max,"+
    "35 nodes rot_avg,35 nodes rot_max,"+
    "40 nodes rot_avg,40 nodes rot_max,"+
    "45 nodes rot_avg,45 nodes rot_max,"+
    "50 nodes rot_avg,50 nodes rot_max,"+"\n")
    for num_nodes2 in range(5, 55, 5):
        outputFile.write(str(num_nodes2) + ",")
        for num_nodes1 in range(5,55,5):
            print(str(num_nodes1) + " node layer...")
            inputLayer = Input(shape=(21,))
            hiddenLayer1 = Dense(num_nodes1)(inputLayer)
            hiddenLayer2 = Dense(num_nodes2)(hiddenLayer1)
            outputLayer = Dense(3)(hiddenLayer2)
            model = Model(inputs=inputLayer, outputs=outputLayer)
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
            history = model.fit(inputDataTrain, outputDataTrain, 32, 2000)

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
