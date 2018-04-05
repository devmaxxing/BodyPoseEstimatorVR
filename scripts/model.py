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
outputLayer = Dense(7)(hiddenLayer1)
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

class TFCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, saver, sess):
        self.saver = saver
        self.sess = sess
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.count += 1
        if self.count == 200:
            self.saver.save(self.sess, 'freeze/checkpoint.ckpt', global_step=epoch)

tf_graph = K.get_session().graph
tf_saver = tf.train.Saver()
tfckptcb = TFCheckpointCallback(tf_saver, K.get_session())

history = model.fit(inputDataTrain, outputDataTrain, 32, 200, callbacks=[tfckptcb])

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
result, avg, maxdiff = e.Difference(outputDataTest, test)
print("max: " + str(maxdiff))
print("sum max: " + str(sum(maxdiff)))
print("avg: " + str(avg))
print("sum avg: " + str(sum(avg)))

#output prediction to specified file based on algorithmic estimator
esti = Estimator()
estimate = esti.Estimate(inputDataTest)
result, avg, maxdiff = e.Difference(estimate, test)

print("max estimate: " + str(maxdiff))
print("sum max estimate: " + str(sum(maxdiff)))
print("avg estimate: " + str(avg))
print("sum avg estimate: " + str(sum(avg)))

if len(sys.argv) > 4:
    outputFile = open(sys.argv[4],'w')
    json.dump(array(estimate).tolist(), outputFile)
    outputFile.close()

#export the model
export_path = "../models"
K.set_learning_phase(0)
config = model.get_config()
weights = model.get_weights()
new_model = Model.from_config(config)
new_model.set_weights(weights)
 
tf.saved_model.simple_save( K.get_session(),
                            export_path,
                            inputs={'input': new_model.inputs[0]},
                            outputs={'output': new_model.outputs[0]})

freeze_graph.freeze_graph(input_graph = export_path +'/saved_model.pb',
              input_binary = True,
              input_checkpoint = './freeze/checkpoint.ckpt',
              output_node_names = "action",
              output_graph = export_path +'/saved_model.bytes' ,
              clear_devices = True, initializer_nodes = "",input_saver = "",
              restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")

 

