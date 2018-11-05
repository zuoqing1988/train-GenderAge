set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_GA_112.py --mode gender_age --lr 0.1 --image_set imdb_celeba_train --prefix model/GA112 --end_epoch 100 --lr_epoch 10,20 --frequent 100 --batch_size 384 
pause 