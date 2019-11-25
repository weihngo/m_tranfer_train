# -*- coding: UTF-8 -*-

import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
from PIL import ImageOps
from forward import forward  # 导入前向网络模块
import argparse  # 导入参数选择模块
import random


#
## 修改以下7个参数以开启训练
#PATH_IMG="./test_imgs/ruanwei.jpeg"  # 参数：选择测试图像
#SELECT_LABEL_NUMS=4  #选择风格数量
#LABEL=(0,1,2,3)  #选择风格
#ALPHA=(0.25,0.25,0.25,0.25)  # 参数：Alpha，风格权重，默认为0.25

# 设置参数
parser = argparse.ArgumentParser()  # 定义一个参数设置器
# 固定参数
parser.add_argument("--LABELS_NUMS", type=int, default=20)  # 参数：风格数量
parser.add_argument("--PATH_MODEL", type=str, default="./model/")  # 参数：模型存储路径
parser.add_argument("--PATH_RESULTS", type=str, default="./results/")  # 参数：测试结果存储路径
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs/")  # 参数：风格图片路径


args = parser.parse_args()  # 定义参数集合args


class Stylizer(object):

    def __init__(self, stylizer_arg,PATH_IMG,SELECT_LABEL_NUMS,LABEL,ALPHA):
        self.stylizer_arg = stylizer_arg
        self.content = tf.placeholder(tf.float32, [1, None, None, 3])  # 图片输入定义
        self.weight = tf.placeholder(tf.float32, [1, stylizer_arg.LABELS_NUMS])  # 风格权重向量，用于存储用户选择的风格
        self.target = forward(self.content, self.weight)  # 定义将要生成的图片
        self.img_path = PATH_IMG
        self.select_label_nums=SELECT_LABEL_NUMS
        self.img = None
        self.label_list = LABEL[0:SELECT_LABEL_NUMS]
        self.alpha_list = ALPHA[0:SELECT_LABEL_NUMS]
        self.input_weight = None
        self.style_weight = None  #stylizer_arg.ALPHA
        self.img = None
        self.img_2 = None

        self.sess = tf.Session()  # 定义一个sess
        self.sess.run(tf.global_variables_initializer())  # 变量初始化
        saver = tf.train.Saver()  # 定义模型saver
        ckpt = tf.train.get_checkpoint_state(stylizer_arg.PATH_MODEL)  # 从模型存储路径中获取模型
        if ckpt and ckpt.model_checkpoint_path:  # 从检查点中恢复模型
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __del__(self):
        self.sess.close()

    def read_image(self):
        # 读取图像
        img_input = Image.open(self.img_path).convert('RGB')
        # 不断对图片进行缩小，直到符合尺寸限制
        # 若图片太大，生成速度将会非常慢，若需生成原图，可将此段注释掉
        while img_input.width * img_input.height > 500000:
            img_input = img_input.resize((int(img_input.width / 1.5), int(img_input.height / 1.5)))
        # 转成数组
        self.img = np.array(img_input)
        
    def set_weight(self, weight_dict):
        # 初始化input_weight
        self.input_weight = np.zeros([1, self.stylizer_arg.LABELS_NUMS])
        # 传入权重字典，键为权重编号，键值为权重，并存到input_weight中
        for k, v in weight_dict.items():
            self.input_weight[0, k] = v

    def set_style_weight(self):
        if self.select_label_nums==1:
            self.style_weight = 1
        else:
            self.style_weight = self.alpha_list
            self.style_weight = self.style_weight/np.sum(self.style_weight)
            self.style_weight = tuple(self.style_weight)

    def stylize(self, alpha_list=None):
        # 将风格列表及对应的权重放入字典中
        if self.select_label_nums==1:
            weight_dict = dict([(self.label_list,alpha_list)])
        else:
            weight_dict = dict(zip(self.label_list, alpha_list))
        self.set_weight(weight_dict)
        # 打印提示信息
        print('Generating', self.img_path.split('/')[-1], '--style:', str(self.label_list),
              '--alpha:', str(alpha_list), )
        # 进行风格融合与迁移
        img = self.sess.run(self.target,
                            feed_dict={self.content: self.img[np.newaxis, :, :, :], self.weight: self.input_weight})
        # 保存单张图片
        # 直接str(tuple)会产生空格，js无法读取
        file_name = self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + str(self.label_list) + '_' + str(alpha_list) + '.jpg'
        file_name = file_name.replace(' ','')
        Image.fromarray(np.uint8(img[0, :, :, :])).save(file_name)
        return img
    
    def generate_single_result(self):
        alphas = self.style_weight
        img_return = self.stylize(alphas)
        img_return = Image.fromarray(np.uint8(img_return[0, :, :, :]))
        width, height = img_return.size
        img_2 = Image.new(img_return.mode, (width * 2, height))
        img_2.paste(
            center_crop_img(Image.open(self.img_path)).resize((width, height)),
            (0, 0, width, height))
        img_2.paste(img_return, (width, 0, width * 2, height))

        self.img_return = img_return
        self.img = img_2
        

    def save_single_result(self):
        self.img_return.save(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + str(
            self.label_list) + '_result_1' + '.jpg')
        self.img_return.show()
        self.img.save(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + str(
            self.label_list) + '_result_1_1' + '.jpg')
        self.img.show()
        print('Image Matrix Saved Successfully!')
    

def center_crop_img(img):
    """
    对图片按中心进行剪裁位正方形
    :param img: 原始图片
    :return:
    """
    width = img.size[0]
    height = img.size[1]
    offset = (width if width < height else height) / 2
    img = img.crop((
        width / 2 - offset,
        height / 2 - offset,
        width / 2 + offset,
        height / 2 + offset
    ))
    return img

def get_image_matrix(PATH_IMG,SELECT_LABEL_NUMS,LABEL,ALPHA):
    # 初始化
    stylizer0 = Stylizer(args,PATH_IMG,SELECT_LABEL_NUMS,LABEL,ALPHA)
    # 读取图像文件
    stylizer0.read_image()
    # 载入权重
    stylizer0.set_style_weight()
    #实用迁移，生成单一图片
    stylizer0.generate_single_result()
    stylizer0.save_single_result()
    # 释放对象
    del stylizer0
    # 重置图
    tf.reset_default_graph()


# 主程序
get_image_matrix(PATH_IMG="./test_imgs/ruanwei.jpeg",SELECT_LABEL_NUMS=5,LABEL=(0,1,2,3,6),ALPHA=(0.25,0.25,0.25,0.25,4))
#PATH_IMG内容图片路径
#SELECT_LABEL_NUMS融合风格总
#LABEL风格代号元组
#ALPHA各风格对应权重



