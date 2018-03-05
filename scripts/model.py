from keras.models import Model
from keras.layers import Input, Dense
from parser import Parser
import sys

a = Input(shape=(21,))
b = Dense(7)(a)
model = Model(inputs=a, outputs=b)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

p = Parser()
dataFile = sys.argv[1]
p.runParse(dataFile)
inputData = p.getInputs()
outputData = p.getOutputs()
model.fit(inputData, outputData)

