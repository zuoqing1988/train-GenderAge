import cv2
import threading
from tools import image_processing
import numpy as np
import math

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.labels = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.labels
        except Exception:
            return None

def get_minibatch_thread(imdb, num_classes, im_size):
    num_images = len(imdb)
    processed_ims = list()
    label = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        h, w, c = im.shape
        cur_label = imdb[i]['label']
        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = image_processing.transform(im)
        processed_ims.append(im_tensor)
        label.append(cur_label)
        
    return processed_ims, label

def get_minibatch(imdb, num_classes, im_size, thread_num = 4):
    # im_size: 112
    #flag = np.random.randint(3,size=1)
    num_images = len(imdb)
    thread_num = max(2,thread_num)
    num_per_thread = math.ceil(num_images/thread_num)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,num_classes,im_size))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    label = list()
   
    
    for t in range(thread_num):
        cur_process_ims, cur_label = threads[t].get_result()
        #print len(cur_process_ims)
        #print len(cur_label)
        processed_ims = processed_ims + cur_process_ims
        label = label + cur_label
        
    im_array = np.vstack(processed_ims)
    label_array = np.array(label)
    
    data = {'data': im_array}
    label = {'label': label_array}
   
    

    return data, label

def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    im = cv2.imread(imdb[0]['image'])
    im_array = im
    data = {'data': im_array}
    label = {}
    return data, label
