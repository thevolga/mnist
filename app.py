import tensorflow as tf
from PIL import Image, ImageFilter
from tensorflow.examples.tutorials.mnist import input_data
from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request
from flask import url_for
from datetime import datetime
import logging
import flask

# from cassandra.cluster import Cluster
# from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

app = Flask(__name__)

model = None
sess = tf.InteractiveSession()
KEYSPACE = "mykeyspace"

def interact_with_DB(pre_res, image_info):
    cluster = Cluster(contact_points = ['127.0.0.1'], port = 9042)
    session = cluster.connect()

    entry_info = ','.join(str(i) for i in image_info)
    number_info = pre_res
    pic = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    date_info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
       session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)

    #    log.info("setting keyspace...")
    #    session.set_keyspace(KEYSPACE)

    except:
        pass
    session.set_keyspace(KEYSPACE)



    # 建表
    try:
        session.execute("""
            CREATE TABLE update_log (
                res int,
                datetime text,
                number text,
                PRIMARY KEY(number)
            )
        
        """)

    except:
        pass

    try:
        session.execute("""
            CREATE TABLE pic_info (
                info text,
                number text,
                PRIMARY KEY(number)
            )
        """)
    except:
        pass

    session.execute("""
            INSERT INTO update_log (res, datetime, number) Values (%s,%s,%s)
    """,
    (number_info, date_info, pic))

    session.execute("""
    		INSERT INTO image (info, number) 
    	        Values (%s,%s) 
    		""",
    (entry_info, pic)
         )


def prepare_image(image): # 处理图片为28*28的灰度图
    # myimage = '1.png'  # 图片路径
    image = image.resize((28, 28))
    image = image.convert('L')  # 转换成灰度图
    tv = list(image.getdata())  # 获取像素值

    # 转换像素范围到[0 1], 0是纯白 1是纯黑
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva

def conv2d(x, W):
      """conv2d returns a 2d convolution layer with full stride."""
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def train_model(res): #mnist_softmax.py
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./SAVE/model.ckpt") #这里使用了之前保存的模型参数
    # print ("Model restored.")

    prediction = tf.argmax(y_conv,1)
    predint = prediction.eval(feed_dict={x: [res],keep_prob: 1.0}, session=sess)
        # print(h_conv2)

        # print('识别结果:')
        # print(predint[0])
    return predint


@app.route("/mnist", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
    	if flask.request.files.get("image"):
            myimage = flask.request.files["image"].read()

            # preprocess the image and prepare it for classification
            res = prepare_image(myimage)
            predint = train_model(res)

            data = predint[0]
            # 与Cassandra交互
            interact_with_DB(data, res)
            data["success"] = True
            
    return flask.jsonify(data)


if __name__=="__main__":
    app.run(port=10009)