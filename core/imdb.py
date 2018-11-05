import mxnet as mx
import os
import cPickle
import numpy as np
from config import config

class IMDB(object):
    def __init__(self, name, im_size, image_set, root_path, dataset_path, mode='train'):
        self.name = name + '_' + image_set
        self.im_size = im_size
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.mode = mode

        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)


    @property
    def cache_path(self):
        """Make a directory to store all caches

        Parameters:
        ----------
        Returns:
        -------
        cache_path: str
            directory to store caches
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path


    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def gt_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                imdb = cPickle.load(f)
            print '{} gt imdb loaded from {}'.format(self.name, cache_file)
            return imdb
        gt_imdb = self.load_annotations()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb


    def image_path_from_index(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """
        image_file = index
        if "." not in image_file:
            image_file = image_file + '.jpg'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self):
        """Load annotations

        Parameters:
        ----------
        Returns:
        -------
        imdb: dict
            image database with annotations
        """
        annotation_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            #print(annotation)
            im_path = self.image_path_from_index(index)
            imdb_ = dict()
            imdb_['image'] = im_path
            if self.mode == 'test':
                pass
            else:
                label = annotation[1:]
                imdb_['label'] = label
                imdb_['flipped'] = False
                
            imdb.append(imdb_)
        return imdb


    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print 'append flipped images to imdb', len(imdb)

        for i in range(len(imdb)):
            imdb_ = imdb[i]
            entry = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'flipped': True}

            imdb.append(entry)
        self.image_set_index *= 2
        return imdb

