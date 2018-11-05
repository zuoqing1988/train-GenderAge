import numpy as np
from easydict import EasyDict as edict

config = edict()
config.root = 'G:/GenderAge'

config.low_age = 10
config.high_age = 80
config.AGE = config.high_age-config.low_age+1

config.GENDER_OHEM = True
config.GENDER_OHEM_RATIO = 0.7
config.AGE_OHEM = False
config.AGE_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [15, 21, 30]
