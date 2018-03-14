from keras.models import Model
from keras.layers import Input, Dense
from util.parser import Parser
from numpy import array
import sys

inputLayer = Input(shape=(21,))
hiddenLayer = Dense(14)(inputLayer)
outputLayer = Dense(7)(hiddenLayer)
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
model.fit(inputDataTrain, outputDataTrain, 1)

inputDataTest = array(p.Parse(dataFileTest))
print(inputDataTest.shape)
outputDataTest = array(p.ParseSpine(dataFileTest))
print(outputDataTest.shape)
loss_and_metrics = model.evaluate(inputDataTest, outputDataTest)
print(loss_and_metrics)

exampleValues = array([inputDataTest[0,:]])
print(exampleValues)
print(model.predict(exampleValues, 1))