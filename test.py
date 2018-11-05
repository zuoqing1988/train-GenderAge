import argparse
import cv2
import sys
import numpy as np
import mxnet as mx
from config import config
import datetime

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    if args.gpu>=0:
      ctx = mx.gpu(args.gpu)
    else:
      ctx = mx.cpu()
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'conv7')


  def get_input(self, face_img):
   
    aligned = np.transpose(face_img, (2,0,1))
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    return db


  def get_ga(self, data):
    self.model.forward(data, is_train=False)
    ret = self.model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    print(g)
    gender = np.argmax(g)
    age_num = config.AGE
    end_id = age_num*2 + 2
    a = ret[:,2:end_id].reshape( (age_num,2) )
    print(a)
    a = np.argmin(a, axis=1)
    age = int(sum(a)) + config.low_age

    return gender, age
	

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='test_data\\00_.jpg', help='')
parser.add_argument('--model', default='model/GA112,100', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
args = parser.parse_args()
model = FaceModel(args)
#img = cv2.imread('test_data\\00_.jpg')
img = cv2.imread(args.image)
img = img - 127.5
img = img / 128.0
img = model.get_input(img)
#f1 = model.get_feature(img)
#print(f1[0:10])

gender, age = model.get_ga(img)
print('gender is',gender)
print('age is', age)

