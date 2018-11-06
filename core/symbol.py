import mxnet as mx
from config import config
import symbol_utils

bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)
    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj
    
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity

base_dim = 16
def GA_Net112(mode, batch_size, test=False):
    """
    #Proposal Network
    #input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")
    
    # data = 112X112
    # conv1 = 56X56
    conv1 = Conv(data, num_filter=base_dim, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
    conv2 = Residual(conv1, num_block=2, num_out= base_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=base_dim, name="res2")
    
	#conv23 = 28X28
    conv23 = DResidual(conv2, num_out=base_dim*2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=base_dim*2, name="dconv23")
    conv3 = Residual(conv23, num_block=6, num_out=base_dim*2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=base_dim*2, name="res3")
    
	#conv34 = 14X14
    conv34 = DResidual(conv3, num_out=base_dim*4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=base_dim*4, name="dconv34")
    conv4 = Residual(conv34, num_block=10, num_out=base_dim*4, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=base_dim*4, name="res4")
    
	#conv45 = 7X7
    conv45 = DResidual(conv4, num_out=base_dim*8, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=base_dim*8, name="dconv45")
    conv5 = Residual(conv45, num_block=2, num_out=base_dim*8, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=base_dim*8, name="res5")
    
	# conv6 = 1x1
    conv6 = Conv(conv5, num_filter=base_dim*8, num_group=base_dim*8, kernel=(7, 7), pad=(0, 0), stride=(1, 1), name="conv6")
    if mode == "gender_age":
        conv7 = mx.symbol.Convolution(data=conv6, kernel=(1, 1), num_filter=config.AGE*2+2, name="conv7")
    else:
        conv7 = mx.symbol.Convolution(data=conv6, kernel=(1, 1), num_filter=2, name="conv7")

    if test:
        group = mx.symbol.Group([conv7])
    else:
        gender_label = mx.symbol.slice_axis(data = label, axis=1, begin=0, end=1)
        gender_label = mx.symbol.reshape(gender_label, shape=(batch_size,))
        gender_fc = mx.symbol.slice_axis(data = conv7, axis=1, begin=0, end=2)
        gender_fc_reshape = mx.symbol.Reshape(data = gender_fc, shape=(-1, 2), name="gender_fc_reshape")
        gender_prob = mx.symbol.SoftmaxOutput(data=gender_fc_reshape, label = gender_label, name='gender_prob', normalization='valid', use_ignore=True, ignore_label=-1)

        outs = [gender_prob]
        if mode == "gender_age":
            for i in range(config.AGE):
                age_label = mx.symbol.slice_axis(data = label, axis=1, begin=i+1, end=i+2)
                age_label = mx.symbol.reshape(age_label, shape=(batch_size,))
                age_fc = mx.symbol.slice_axis(data = conv7, axis=1, begin=2+i*2, end=4+i*2)
                age_fc_reshape = mx.symbol.Reshape(data = age_fc, shape=(-1, 2), name="age_fc_reshape_%d"%i)
                age_prob = mx.symbol.SoftmaxOutput(data=age_fc_reshape, label = age_label, name='age_prob_%d'%i, normalization='valid', 
                    use_ignore=True, ignore_label=-1, grad_scale=1)
                outs.append(age_prob)
        outs.append(mx.sym.BlockGrad(conv7))
        group = mx.symbol.Group(outs)
    return group
