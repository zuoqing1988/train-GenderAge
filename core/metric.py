import mxnet as mx
import numpy as np
from config import config


class GenderAccuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(GenderAccuracy, self).__init__('GenderAccuracy')

    def update(self, labels, preds):
        #print(preds[0])
        batch_size = labels[0].shape[0]
        gender_label = labels[0][:,0]
        gender_prob = preds[0]
        gender_prob = mx.ndarray.argmax_channel(gender_prob).asnumpy().astype('int32')
        gender_label = gender_label.asnumpy().astype('int32')
        #print(gender_label)
        #print(gender_prob)
        gender_valid = gender_label >-1
        #print(gender_valid)
        gender_good = gender_prob == gender_label
        real_good = gender_valid.astype('int32')*gender_good.astype('int32')
        #print(real_good)
        self.sum_metric += real_good.sum()

        self.num_inst += gender_valid.sum()


class GenderLogLoss(mx.metric.EvalMetric):
    def __init__(self):
        super(GenderLogLoss, self).__init__('GenderLogLoss')

    def update(self, labels, preds):
        batch_size = labels[0].shape[0]
        gender_prob = preds[0].asnumpy()
        gender_label = labels[0][:,0]
        gender_label = gender_label.asnumpy().astype('int32')
        gender_valid = gender_label >-1
       
        gender_prob = gender_prob.reshape(-1, 2)
        gender_prob_keep = [gender_prob[i,:] for i in range(batch_size) if gender_valid[i] == 1]
        gender_label_keep = [gender_label[i] for i in range(batch_size) if gender_valid[i] == 1]
        keep_num = len(gender_label_keep)
        gender_label_keep = np.array(gender_label_keep)
        gender_prob_keep = np.array(gender_prob_keep)
        gender = gender_prob_keep[np.arange(keep_num), gender_label_keep.flat]

        gender += config.EPS
        gender_loss = -1 * np.log(gender)

        gender_loss = np.sum(gender_loss)
        self.sum_metric += gender_loss
        self.num_inst += keep_num


class AGE_MAE(mx.metric.EvalMetric):
    def __init__(self):
        super(AGE_MAE, self).__init__('AGE_MAE')
   
    def update(self,labels, preds):
        #print(labels[0].shape)
        #print(preds[1])
        batch_size = labels[0].shape[0]
        age_truth = np.count_nonzero(labels[0][:,1:].asnumpy().astype('int32'), axis=1)
        age_valid = (labels[0][:,1].asnumpy().astype('int32') > -1).astype('int32')
        age_pred = np.zeros(batch_size)
        for i in range(config.AGE):
            age_prob = preds[i+1]
            age_prob = mx.ndarray.argmax_channel(age_prob).asnumpy().astype('int32')
            age_pred += age_prob.flat
              
        e = np.abs(age_truth - age_pred)
        e = e * age_valid
        error = np.sum(e.flat)
        self.sum_metric += error
        self.num_inst += np.count_nonzero(age_valid)
