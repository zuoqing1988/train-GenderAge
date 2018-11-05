train-GenderAge

此项目在win 7系统上开发，其他环境未测试

**1.数据准备**

从[此链接](https://pan.baidu.com/s/1s8amSxHYHQQzO4nkjBo9Ww)下载训练这个工程所需要的数据，解压到data文件夹里面

解压之后的目录结构为

data\celeba-112X112

data\imdb-112X112

data\wiki-112X112

data\imdb_celeba_train.txt

data\imdb_train.txt

data\wiki_train.txt


**2.训练Gender and Age**

双击GA112_train.bat开始训练

**3.仅训练Gender**

打开GA112_train.bat, 把--mode gender_age 改成--mode gender 

注意：这时候直接运行test.bat会崩的，需要你手工注释掉test.py里age相关的代码

**4.导出成ZQCNN格式**

使用默认的网络格式，ZQCNN上单线程约7ms一次。

请学习[ZQCNN](https://github.com/zuoqing1988/ZQCNN-v0.0)，然后用mxnet2zqcnn导出（需要手工修改一些东西）