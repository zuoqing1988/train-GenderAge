import argparse
import mxnet as mx
import sys,os
sys.path.append(os.getcwd())
from core.imdb import IMDB
from train import train_net
from core.symbol import GA_Net112

def train_GA_net(mode, image_set, root_path, dataset_path, prefix, ctx,
                pretrained, epoch, begin_epoch, end_epoch, batch_size, thread_num,
                frequent, lr, lr_epoch, resume):
    imdb = IMDB("GA", 112, image_set, root_path, dataset_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)
    sym = GA_Net112(mode,batch_size)

    train_net(mode,sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, gt_imdb, batch_size, thread_num, 112,
              112, frequent, not resume, lr, lr_epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net(112-net)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', dest='mode', help='gender or gender_age',
                        default='gender_age', type=str)
    parser.add_argument('--image_set', dest='image_set', help='training set',
                        default='imdb_train', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='data', type=str)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default='model/GA112', type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained prefix',
                        default='model/GA112', type=str)
    parser.add_argument('--epoch', dest='epoch', help='load epoch',
                        default=0, type=int)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size of training',
                        default=128, type=int)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num of training',
                        default=4, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_epoch', dest='lr_epoch', help='learning rate epoch',
                        default='8,14', type=str)
    parser.add_argument('--resume', dest='resume', help='continue training', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    lr_epoch = [int(i) for i in args.lr_epoch.split(',')]
    train_GA_net(args.mode, args.image_set, args.root_path, args.dataset_path, args.prefix, ctx,
                args.pretrained, args.epoch,
                args.begin_epoch, args.end_epoch, args.batch_size, args.thread_num, args.frequent, args.lr, lr_epoch, args.resume)
