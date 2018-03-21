from keras.models import Model
from keras.layers import Input, Dense
from util.parser import Parser
from util.Evaluator import Evaluator
from numpy import array
import sys
import json
import matplotlib.pyplot as plt

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
history = model.fit(inputDataTrain, outputDataTrain, 32, 200)

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

inputDataTest = array(p.Parse(dataFileTest))
print(inputDataTest.shape)
outputDataTest = array(p.ParseSpine(dataFileTest))
print(outputDataTest.shape)
loss_and_metrics = model.evaluate(inputDataTest, outputDataTest)
print(loss_and_metrics)

#test the model
test = model.predict(inputDataTest, 1)

#output prediction if output file is specified
if len(sys.argv) > 3:
    outputFile = open(sys.argv[3],'w')
    json.dump(array(test).tolist(), outputFile)
    outputFile.close()

#calculate differences
e = Evaluator()
result, avg = e.Difference(outputDataTest, test)
print("result is:")
print(result)
print("avg is:")
print(avg)
