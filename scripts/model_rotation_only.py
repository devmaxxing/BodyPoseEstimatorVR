from keras.models import Model
from keras.layers import Input, Dense
from util.parser import Parser
from util.Evaluator import Evaluator
from util.Estimator import Estimator
from numpy import array
import sys
import json
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import builder as saved_model_builder

def weighted_mean_squared_error(y_true, y_pred):
    difference = y_pred - y_true
    weights = array([20, 20, 20, 20, 1, 1, 1])
    return K.mean(K.square((difference*weights)), axis=-1)

inputLayer = Input(shape=(21,))
hiddenLayer1 = Dense(11)(inputLayer)
outputLayer = Dense(4)(hiddenLayer1)
model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

p = Parser()
dataFileTrain = sys.argv[1]
dataFileTest = sys.argv[2]

inputDataTrain = array(p.Parse(dataFileTrain))
outputDataTrain = array(p.ParseSpineRotation(dataFileTrain))

history = model.fit(inputDataTrain, outputDataTrain, 32, 2000)

# summarize history for loss
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#export the model
# export_path = "freeze_rot"
# tf.train.Saver().save(K.get_session(), export_path + '/checkpoint.ckpt')

# tf.train.write_graph(K.get_session().graph.as_graph_def(),
#                      export_path, 'graph.pbtxt', as_text=True)
# tf.train.write_graph(K.get_session().graph.as_graph_def(),
#                      export_path, 'graph.pb', as_text=False)

# freeze_graph.freeze_graph(input_graph = export_path +'/graph.pbtxt',
#               input_binary = False,
#               input_checkpoint = export_path + '/checkpoint.ckpt',
#               output_node_names = "dense_2/BiasAdd",
#               output_graph = export_path +'/model.bytes' ,
#               clear_devices = True, initializer_nodes = "",input_saver = "",
#               restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")


inputDataTest = array(p.Parse(dataFileTest))
outputDataTest = array(p.ParseSpineRotation(dataFileTest))
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
result, avg, maxdiff = e.Difference(outputDataTest, test)
print("max: " + str(maxdiff))
print("sum max: " + str(sum(maxdiff)))
print("avg: " + str(avg))
print("sum avg: " + str(sum(avg)))

#output prediction to specified file based on algorithmic estimator
esti = Estimator()
estimate = esti.Estimate(inputDataTest)
result, avg, maxdiff = e.Difference(outputDataTest, estimate)

print("max estimate: " + str(maxdiff))
print("sum max estimate: " + str(sum(maxdiff)))
print("avg estimate: " + str(avg))
print("sum avg estimate: " + str(sum(avg)))

if len(sys.argv) > 4:
    outputFile = open(sys.argv[4],'w')
    json.dump(array(estimate).tolist(), outputFile)
    outputFile.close()



 

