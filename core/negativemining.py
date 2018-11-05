import mxnet as mx
import numpy as np
from config import config

class NegativeMiningOperator(mx.operator.CustomOp):
    def __init__(self, gender_ohem=config.GENDER_OHEM, gender_ohem_ratio=config.GENDER_OHEM_RATIO,
            age_ohem=config.AGE_OHEM, age_ohem_ratio=config.AGE_OHEM_RATIO):
        super(NegativeMiningOperator, self).__init__()
        self.gender_ohem = gender_ohem
        self.gender_ohem_ratio = gender_ohem_ratio
        self.age_ohem = age_ohem
        self.age_ohem_ratio = age_ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        gender_prob = in_data[0][0].asnumpy() # batchsize x 2 x 1 x 1
        gender_label = in_data[1][0].asnumpy().astype(int) # batchsize x 1
        for i in range(config.AGE):
            age_prob = in_data[0][i+1].asnumpy() # batchsize x 81
            age_label = in_data[1][i+1].asnumpy().astype(int) # batchsize x 1

        #print(gender_prob)
        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], in_data[1])

        # gender
        gender_prob = gender_prob.reshape(-1, 2)
        valid_inds = np.where(label > -1)[0]
        gender_keep = np.zeros(gender_prob.shape[0])

        if self.gender_ohem:
            keep_num = int(len(valid_inds) * self.gender_ohem_ratio)
            gender_valid = gender_prob[valid_inds, :]
            label_valid = label.flatten()[valid_inds]

            gender = gender_valid[np.arange(len(valid_inds)), label_valid] + config.EPS
            log_loss = - np.log(gender)
            keep = np.argsort(log_loss)[::-1][:keep_num]
            gender_keep[valid_inds[keep]] = 1
        else:
            gender_keep[valid_inds] = 1
        self.assign(out_data[2], req[2], mx.nd.array(gender_keep))

        # age
        valid_inds = np.where(abs(label) == 1)[0]
        age_keep = np.zeros(gender_prob.shape[0])

        if self.age_ohem:
            keep_num = int(len(valid_inds) * self.age_ohem_ratio)
            age_valid = age_pred[valid_inds, :]
            age_target_valid = age_target[valid_inds, :]
            square_error = np.sum((age_valid - age_target_valid)**2, axis=1)
            keep = np.argsort(square_error)[::-1][:keep_num]
            age_keep[valid_inds[keep]] = 1
        else:
            age_keep[valid_inds] = 1
        self.assign(out_data[3], req[3], mx.nd.array(age_keep))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        gender_keep = out_data[2].asnumpy().reshape(-1, 1)
        age_keep = out_data[3].asnumpy().reshape(-1, 1)

        gender_grad = np.repeat(gender_keep, 2, axis=1)
        age_grad = np.repeat(age_keep, 81, axis=1)

        gender_grad /= len(np.where(gender_keep == 1)[0])
        age_grad /= len(np.where(age_keep == 1)[0])

        gender_grad = gender_grad.reshape(in_data[0].shape)
        self.assign(in_grad[0], req[0], mx.nd.array(gender_grad))
        self.assign(in_grad[1], req[1], mx.nd.array(age_grad))


@mx.operator.register("negativemining")
class NegativeMiningProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['gender_prob', 'age_pred', 'label', 'age_target']

    def list_outputs(self):
        return ['gender_out', 'age_out', 'gender_keep', 'age_keep']

    def infer_shape(self, in_shape):
        keep_shape = (in_shape[0][0], )
        return in_shape, [in_shape[0], in_shape[1], keep_shape, keep_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator()
