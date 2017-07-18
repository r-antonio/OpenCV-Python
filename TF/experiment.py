import tensorflow as tf
import numpy as np
import os,sys
from sklearn.metrics import confusion_matrix
from time import gmtime, strftime

sys.path.insert(0, os.getcwd())

from tflearn.data_utils import shuffle, to_categorical
import utils
from dataload.LSA16h import LSA16h
from dataload.PH import PH
from dataload.Dataset import Dataset
from dataload.BatchedDataset import BatchedDataset

import model
import tflearn

def write_dict(d, filename):
  with open(filename, "a") as input_file:
    sorted_keys=sorted(d.keys())
    for k in sorted_keys:
      v=d[k]
      line = '{}={}'.format(k, v)
      print(line, file=input_file)



def get_experiment_folder(base_folder):
  def make_path(index):
    name= base_name + str(index).zfill(digits)
    return os.path.join(base_folder,name)
  digits=4
  base_name="exp"
  files = [f for f in os.listdir(base_folder) if f.startswith(base_name) and (not os.path.isfile(f))]
  files.sort()
  if files:
    last_experiment=files[-1]
    index=int(last_experiment[-digits:])
    return make_path(index+1)
  else:
    return make_path(0)

def save_results(experiment_folder):
  pass

def get_lsadataset():
  dataset='lsa32x32/nr/gray'
  input_folderpath='../datasets/%s' % dataset
  lsa=LSA16h(dataset,input_folderpath)
  lsa.x=list(map(lambda i: i.astype(float)/255.0,lsa.x))
  d=lsa.as_dataset()
  return d
def get_phdataset():
  dataset='ph32'
  input_folderpath='../datasets/%s' % dataset
  lsa=PH(dataset,input_folderpath)
  lsa.x=list(map(lambda i: i.astype(float)/255.0,lsa.x))
  d=lsa.as_dataset()
  return d

def get_dataset():
  return get_lsadataset()

def evaluate(model,train_dataset,test_dataset,experiment_folder,parameters):
  train_y= np.argmax(model.predict(train_dataset.x),axis=1)
  train_accuracy=np.mean(train_y==train_dataset.y)
  test_y= np.argmax(model.predict(test_dataset.x),axis=1)
  test_accuracy=np.mean(test_y==test_dataset.y)
  print("Train accuracy: %f" % train_accuracy)
  print("Test accuracy: %f" % test_accuracy)
  parameters["accuracy_train"]=train_accuracy
  parameters["accuracy_test"]=test_accuracy
  write_dict(parameters,os.path.join(experiment_folder,"results.txt"))
  test_confusion_matrix=confusion_matrix(test_dataset.y, test_y)
  utils.save_confusion_matrix(os.path.join(experiment_folder,"test_confusion.png"),test_confusion_matrix,range(classes))
  train_confusion_matrix=confusion_matrix(train_dataset.y, train_y)
  utils.save_confusion_matrix(os.path.join(experiment_folder,"train_confusion.png"),train_confusion_matrix,range(classes))
  # print("Saving model..")
  # model.save('trained_models/vgg16')

def train(m,train_dataset,parameters,experiment_folder):
  regression = tflearn.regression(m.graph, optimizer='adam', loss='categorical_crossentropy',learning_rate=parameters['learning_rate'])
  checkpoint_path=os.path.join(experiment_folder, 'checkpoint')
  tensorboard_dir= os.path.join(experiment_folder, 'tensorboard')
  model = tflearn.DNN(regression, checkpoint_path=checkpoint_path,
  max_checkpoints=3, tensorboard_verbose=2,tensorboard_dir=tensorboard_dir)
  model.fit(train_dataset.x, train_dataset.y_one_hot, n_epoch=parameters['epochs'], validation_set=0.1, shuffle=True,show_metric=True, batch_size=parameters['batch_size'], snapshot_step=200,snapshot_epoch=False,  run_id='tflearn_snapshots/model_finetuning')
  return model

def save_weights(model,m,experiment_folder):
  for l in m.layers:
    variables=tflearn.get_layer_variables_by_name(l)
    for v in variables:
      w=model.get_weights(v)
      

print("Loading data...")
d=get_dataset()
timestamp=strftime("%Y%m%d_%H:%M:%S", gmtime())
parameters={
            'batch_size':32,
            'epochs':75,
            'learning_rate':0.0007,
            'split':0.8,
            'timestamp':timestamp,
            'dataset':d.id,
            }


classes=max(d.y)+1
image_size=d.x[0].shape
print("Samples in dataset: %d" % len(d.y))
train_dataset,test_dataset=d.split_stratified(parameters['split'])
print("Samples in train dataset: %d" % len(train_dataset.y))
print("Samples in test dataset: %d" % len(test_dataset.y))
# print(train_dataset.y)
# print(classes)
train_dataset.y_one_hot=to_categorical(train_dataset.y,classes)
test_dataset.y_one_hot=to_categorical(test_dataset.y,classes)


input_size=np.prod(image_size)
print("Preparing model...")
with tf.Graph().as_default():
  #s =  tf.InteractiveSession()
  x = tf.placeholder(tf.float32, shape=[ None,image_size[0],image_size[1],image_size[2]])
  y = tf.placeholder(tf.int32, shape=[None])
  #m = model.vgg16(classes,x)
  m = model.inception(classes,x)
  #m =model.simple_feed(classes,x)
  #m =model.two_layers_feed(classes,x)
  #m =model.conv_simple(classes,x)

  #m = model.all_convolutional(classes,x)

  # print("Loading weights..")
  # model.load("/home/facuq/dev/models/vgg16.tflearn")
  parameters["model"]=m.id

  experiments_folder='tmp/'
  experiment_folder=get_experiment_folder(experiments_folder)
  os.mkdir(experiment_folder)
  write_dict(parameters,os.path.join(experiment_folder,"parameters.txt"))
  print("Fitting model "+m.id+"...")
  model=train(m,train_dataset,parameters,experiment_folder)

  print("Finished training.")
  print("Evaluating...")
  evaluate(model,train_dataset,test_dataset,experiment_folder,parameters)
  save_weights(model,m,experiment_folder)
