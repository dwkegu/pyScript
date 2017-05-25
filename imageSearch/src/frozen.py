import tensorflow as tf
from tensorflow.contrib import slim
from imageSearch.src.model.inception_v4 import *
from imageSearch.src import freeze_graph
import os.path

checkpoint_state_name = "checkpoint_state"
input_graph_name = "InceptionV4t.pb"
output_graph_name = "InceptionV4cpp.pb"
input_graph_path = os.path.join("D:/tmp/modelGraph/", input_graph_name)
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = os.path.join("D:/tmp/tensorflow/inception_v4_2016_09_09","inception_v4.ckpt")
output_node_names = "InceptionV4/Logits/PreLogitsFlatten/Reshape," \
                    "InceptionV4/Logits/Logits/BiasAdd," \
                    "InceptionV4/Logits/top_K," \
                    "InceptionV4/Logits/class"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join("E:/tmp/", output_graph_name)
clear_devices = False
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name,output_graph_path,
                          clear_devices, "")

'''
graph = tf.Graph()
graph.as_default()
sess = tf.Session(graph= tf.get_default_graph())
image_size = 299
input_holder = tf.placeholder(tf.float32, shape=[1, image_size, image_size, 3])
arg_scope = inception_v4_arg_scope(weight_decay=0.0)
inception_v4.default_image_size = image_size
with slim.arg_scope(arg_scope):
    logits, end = inception_v4(input_holder, 1001, is_training=False)
    #sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #将模型写成文件，供C++和java使用
    tf.train.write_graph(sess.graph_def, '/tmp/modelGraph', 'InceptionV4text.pb', as_text=False)
'''