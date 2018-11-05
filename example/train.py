import logging
import mxnet as mx
import core.metric as metric
from mxnet.module.module import Module
from core.loader import ImageLoader
from core.imdb import IMDB
from config import config
from tools.load_model import load_param

def train_net(mode, sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, imdb, batch_size, thread_num, im_size,
              net=112, frequent=50, initialize=True, base_lr=0.01, lr_epoch = [6,14]):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    train_data = ImageLoader(imdb, net, batch_size, thread_num, shuffle=True, ctx=ctx)

    if not initialize:
        args, auxs = load_param(pretrained, epoch, convert=True)

    if initialize:
        print "init weights and bias:"
        data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
        print(data_shape_dict)
        arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
        #print(arg_shape)
        #print(aux_shape)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
        init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        args = dict()
        auxs = dict()
        #print 'hello3'
		
        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue

            #print 'init', k

            args[k] = mx.nd.zeros(arg_shape_dict[k])
            init(k, args[k])
            if k.startswith('fc'):
                args[k][:] /= 10


        for k in sym.list_auxiliary_states():
            auxs[k] = mx.nd.zeros(aux_shape_dict[k])
            #print aux_shape_dict[k]
            init(k, auxs[k])

    lr_factor = 0.1
    #lr_epoch = config.LR_EPOCH
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(imdb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch', lr_epoch, 'lr_epoch_diff', lr_epoch_diff
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    eval_metrics = mx.metric.CompositeEvalMetric()
    
    metric1 = metric.GenderAccuracy()
    metric2 = metric.GenderLogLoss()
    if mode == "gender_age":
        metric3 = metric.AGE_MAE()
        for child_metric in [metric1, metric2, metric3]:
            eval_metrics.add(child_metric)
    else:
        for child_metric in [metric1, metric2]:
            eval_metrics.add(child_metric)
    #eval_metrics = mx.metric.CompositeEvalMetric([metric.AccMetric(), metric.MAEMetric(), metric.CUMMetric()])
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.00001,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0}

    mod = Module(sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=end_epoch)

