import tensorflow as tf
from .model.inception_v4 import inception_v4
from .model.inception_v4 import inception_v4_arg_scope
from .PreProcessing import inception_preprocessing
import re
import os
import shutil
import numpy as np
import csv
import argparse

slim = tf.contrib.slim


class ImageNet:
    def __init__(self):
        self._sep = '/'
        self._hasInitModel = False
        self._FLAGS = None
        self._default_num_class = 1001
        self._initModel()

    def _initModel(self):
        self.setFlag()
        self._graph = tf.Graph()
        self._graph.as_default()
        tf_global_step = slim.get_or_create_global_step
        self._sess = tf.Session(graph=tf.get_default_graph())
        self._image_size = self._FLAGS.eval_image_size or inception_v4.default_image_size
        self._input_holder = tf.placeholder(tf.float32, shape=[1, self._image_size, self._image_size, 3])
        arg_scope = inception_v4_arg_scope(weight_decay=0.0)
        inception_v4.default_image_size = self._image_size
        with slim.arg_scope(arg_scope):
            self._logits, self._end = inception_v4(self._input_holder, self._default_num_class, is_training=False)
            self._sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # 将模型写成文件，供C++使用
            # tf.train.write_graph(sess.graph_def, '/tmp/modelGraph', 'InceptionV4', as_text=False)
            # check whether  the FLAGS.checkpoint_path is a directory
            if tf.gfile.IsDirectory(self._FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(self._FLAGS.checkpoint_path)
            else:
                checkpoint_path = self._FLAGS.checkpoint_path
            saver.restore(self._sess, checkpoint_path)
        self._hasInitModel = True

    def _read_images_from_disk(self, file_name):
        """从磁盘读取图片文件
        Args:
          file_name: 图片文件路径 tensor 变量，string类型
        Returns:
          文件内容
        """
        file_content = tf.read_file(file_name)
        try:
            if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
                example = tf.image.decode_jpeg(file_content, 3)
            else:
                example = tf.image.decode_png(file_content, 3)
        except tf.errors.InvalidArgumentError:
            return None
        except IOError:
            return None
        except:
            return None
        return example

    def _getHashKey(self, feature, keyNum=1536):
        _feature = np.reshape(feature, [keyNum])
        mean = np.average(_feature, 0)
        print(mean)
        res = []
        for item in _feature:
            if item > mean:
                res.append(1)
            else:
                res.append(0)
        return res

    def _run(self, operation=0):
        '''

        :param operation: 0 表示仅仅为了获取哈希码，1表示获取哈希码，进行图片归类储存，2表示获取哈希码并储存
        :return: 
        '''
        files = []
        if operation == 0:
            files = [self._FLAGS.image_filename, ]
        elif operation == 1:
            self._save_dir = os.path.expanduser(self._FLAGS.image_class_dir)
            if not os.path.exists(self._save_dir):
                os.mkdir(self._save_dir)
            files = os.listdir(self._FLAGS.image_filename)
        elif operation == 2:
            files = os.listdir(self._FLAGS.image_filename)
        ressult = []
        for file in files:
            if re.match(r'(^.*(?:\.png|\.jpg|\.jpeg)$)', file) is None:
                print("%s is not supported image type" % file)
                continue
            # convert filename to tensor
            file_name = os.path.join(self._FLAGS.image_filename + '/', file)
            print(file_name)
            image = self._read_images_from_disk(file_name)
            if image is None:
                continue
            print("read image success")
            image = inception_preprocessing.preprocess_image(image, self._image_size, self._image_size,
                                                             is_training=False)
            image = tf.reshape(image, [1, self._image_size, self._image_size, 3])
            # _e 包含每一层的输出
            out, _e = self._sess.run([self._logits, self._end],
                                     feed_dict={self._input_holder: image.eval(session=self._sess)})
            topK = np.reshape(_e['top_K'], [-1])
            kind = np.reshape(_e['class'], [-1])
            hashCode1 = self._getHashKey(_e['PreLogitsFlatten'])
            hashCode2 = self._getHashKey(out, 1001)
            if operation == 1:
                spath = os.path.join(self._save_dir + os.sep, str(kind))
                if not os.path.exists(spath):
                    os.mkdir(spath)
                row = [os.path.join(spath + self._sep, file), hashCode1, hashCode2]
                if not os.path.exists(os.path.join(spath + os.sep, 'hashCode.csv')):
                    with open(os.path.join(spath + os.sep, 'hashCode.csv'), mode='w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)
                else:
                    with open(os.path.join(spath + os.sep, 'hashCode.csv'), mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)
                if not os.path.exists(os.path.join(spath + self._sep, file)):
                    shutil.move(os.path.join(self._FLAGS.image_filename + os.sep, file), spath)
                else:
                    os.remove(os.path.join(self._FLAGS.image_filename + os.sep, file))
            elif operation == 2:
                # 找到文件所在目录
                # print(file_name)
                spath = self._FLAGS.image_filename
                row = [os.path.join(spath + self._sep, file), hashCode1, hashCode2]
                if not os.path.exists(os.path.join(spath + self._sep, 'hashCode.csv')):
                    with open(os.path.join(spath + os.sep, 'hashCode.csv'), mode='w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)
                else:
                    with open(os.path.join(spath + os.sep, 'hashCode.csv'), mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)
                if not os.path.exists(os.path.join(spath + os.sep, file)):
                    shutil.move(os.path.join(self._FLAGS.image_filename + os.sep, file), spath)
            else:
                ressult = {"topK": topK, "kind": kind, "hc1": hashCode1, "hc2": hashCode2,
                           "PreLogitsFlatten": _e['PreLogitsFlatten'], "class": out}
        return ressult

    def setFlag(self, batch_size=1, checkpoint_path=None,
                image_class_dir=None, image_filename=None, operation=0, **kwargs):
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()
        tf.app.flags.DEFINE_integer(
            "batch_size", batch_size,
            "evalation batch size")
        tf.app.flags.DEFINE_integer(
            'max_num_batches', None,
            'Max number of batches to evaluate by default use all.')

        tf.app.flags.DEFINE_string(
            'master', '', 'The address of the TensorFlow master to use.')

        tf.app.flags.DEFINE_string(
            'checkpoint_path', 'D:/tmp/tensorflow/inception_v4_2016_09_09/inception_v4.ckpt',
            'The directory where the model was written to or an absolute path to a '
            'checkpoint file.')

        tf.app.flags.DEFINE_string(
            'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

        tf.app.flags.DEFINE_integer(
            'num_preprocessing_threads', 4,
            'The number of threads used to create the batches.')
        if image_class_dir == None:
            tf.app.flags.DEFINE_string("image_class_dir", "/tmp/image", "classified res storing location")
        else:
            tf.app.flags.DEFINE_string("image_class_dir", image_class_dir, "classified res storing location")
        if image_filename == None:
            tf.app.flags.DEFINE_string("image_filename", "/tmp/psf/ImageSearch", "eval image path")
        else:
            tf.app.flags.DEFINE_string("image_filename", image_filename, "eval image path")
        tf.app.flags.DEFINE_string(
            'dataset_name', 'imagenet', 'The name of the dataset to load.')

        tf.app.flags.DEFINE_string(
            'dataset_split_name', 'test', 'The name of the train/test split.')

        tf.app.flags.DEFINE_string(
            'dataset_dir', None, 'The directory where the dataset files are stored.')

        tf.app.flags.DEFINE_integer(
            'labels_offset', 0,
            'An offset for the labels in the dataset. This flag is primarily used to '
            'evaluate the VGG and ResNet architectures which do not use a background '
            'class for the ImageNet dataset.')

        tf.app.flags.DEFINE_string(
            'model_name', 'inception_v4', 'The name of the architecture to evaluate.')

        tf.app.flags.DEFINE_string(
            'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                        'as `None`, then the model_name flag is used.')

        tf.app.flags.DEFINE_float(
            'moving_average_decay', None,
            'The decay to use for the moving average.'
            'If left as None, then moving averages are not used.')

        tf.app.flags.DEFINE_integer(
            'eval_image_size', None, 'Eval image size')
        self._FLAGS = tf.app.flags.FLAGS

    def test(self, batch_size=1, checkpoint_path=None,
             image_class_dir=None, image_filename=None, operation=0, **kwargs):
        '''
        测试图片\n
        :param batch_size: 一次测试的图片数量，默认一张 \n
        :param checkpoint_path: 参数保存位置
        :param image_class_dir: 分类后的图片储存位置
        :param image_filename: 原图片位置
        :return: 哈希码和分类信息
        '''
        self.setFlag(batch_size, checkpoint_path, image_class_dir, image_filename, operation=operation)
        # print(self._FLAGS)
        if not self._hasInitModel:
            self._initModel()
        return self._run(operation=operation)


