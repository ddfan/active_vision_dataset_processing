import os
import tensorflow as tf
import numpy as np
import mobilenet_v1
import cv2 as cv
import matplotlib.pyplot as plt

MODEL_NAME = './checkpoints/mobilenet_v1_1.0_224'

img_size=224
factor=1.0
num_classes=1001
is_training=False
weight_decay = 0.0

tf.reset_default_graph()

inp = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 3),name="input")
arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
with tf.contrib.slim.arg_scope(arg_scope):
	logits, _ = mobilenet_v1.mobilenet_v1(inp,num_classes=num_classes,is_training=is_training,depth_multiplier=factor)

#predictions = tf.contrib.layers.softmax(logits)
#output = tf.identity(predictions, name='output')
output = tf.get_default_graph().get_tensor_by_name("MobilenetV1/Logits/AvgPool_1a/AvgPool:0")
rest_var = tf.contrib.slim.get_variables_to_restore()
var_dict={}
for var in rest_var:
	noscope_name=var.name.replace(':0','')
	var_dict[noscope_name]=var	
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_dict)
saver.restore(sess, MODEL_NAME+'.ckpt')
sess.run(tf.global_variables_initializer())
#for var in tf.trainable_variables():
#	print(var.name)

#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

#test things out with an image
#img=cv.imread("ActiveVisionDataset/Home_001_1/jpg_rgb/000110000010101.jpg",flags=-1)
img=cv.imread("cat.jpg",flags=-1)
#print(img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)[:,:,0:3]
#rgb_image=img[0:1080,420:1500,:]
#print(img.shape)
img=cv.resize(img,(img_size,img_size))
x=200
y=400
#img=rgb_image[x:x+img_size,y:y+img_size,:]
fig,ax = plt.subplots(2)
plt.cla()
ax[0].imshow(img)

img=np.expand_dims(img,axis=0)

out = sess.run(output, feed_dict={inp:img})
out=np.squeeze(out)
print(out, out.shape)
print(np.min(out),np.max(out))

ax[1].plot(np.arange(0,out.size,1),out)
plt.draw()
plt.show()
