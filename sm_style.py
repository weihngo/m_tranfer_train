# -*- coding: UTF-8 -*-

import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
from PIL import ImageOps
from forward import forward  # 导入前向网络模块
import argparse  # 导入参数选择模块
import random
import cv2


def stylize(img,label_list,sess,alpha_list=(1,1,1,1)):
    graph = tf.compat.v1.get_default_graph()
    content = graph.get_tensor_by_name('content:0')
    weight = graph.get_tensor_by_name('weight:0')
    target = graph.get_tensor_by_name('target:0')
    style_num=len(label_list)
    alpha_list=alpha_list[0:style_num]
    input_weight=np.zeros([1,20])
    if style_num==1:
        weight_dict = dict([(label_list,alpha_list)])
    else:
        weight_dict = dict(zip(label_list, alpha_list))
    for k,v in weight_dict.items():
        input_weight[0,k]=v
    # 打印提示信息
    print('creating photo now.please wait')
    # 进行风格融合与迁移
    img = sess.run(target,
                        feed_dict={content: img[np.newaxis, :, :, :], weight: input_weight})
    # 保存单张图片
    # 直接str(tuple)会产生空格，js无法读取
    img=np.uint8(img[0, :, :, :])
    
    return img

def loading_model():
    #预定义模型变量，分配空间
    #tf.reset_default_graph()
    #content = tf.placeholder(tf.float32, [1, None, None, 3])  # 图片输入定义
    #weight = tf.placeholder(tf.float32, [1, 20])  # 风格权重向量，用于存储用户选择的风格
    #target = tf.placeholder(tf.float32, [1, None, None, 3])  # 定义将要生成的图片
    #从恢复点中载入模型
    model_id="./model/"
#    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#    sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # 变量初始化
    ckpt = tf.train.get_checkpoint_state(model_id)  # 从模型存储路径中获取模型
    #saver = tf.train.Saver()  # 定义模型saver
    saver = tf.train.import_meta_graph(os.path.join(model_id, 'model-49901.meta'))
    if ckpt and ckpt.model_checkpoint_path:  # 从检查点中恢复模型
        saver.restore(sess, ckpt.model_checkpoint_path)
    return sess


img=Image.open('./test_imgs/ruanwei.jpeg').convert('RGB')
img=np.array(img)
sess=loading_model()
img=stylize(img,(6,7),sess)
Image.fromarray(img).show()
#img:内容图片矩阵
#6：风格图片代号




