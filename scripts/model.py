from keras.models import Model
from keras.layers import Input, Dense
from util.parser import Parser
from util.Evaluator import Evaluator
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

exampleValues = array([inputDataTest[1,:]])
print(exampleValues)
print(model.predict(exampleValues, 1))

e = Evaluator()
test = model.predict(inputDataTest, 1)
result, avg = e.Difference(outputDataTest, test)
print("result is:")
print(result)
print("avg is:")
print(avg)
