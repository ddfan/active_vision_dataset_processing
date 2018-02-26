import os
import tensorflow as tf
import numpy as np
import mobilenet_v1
import cv2 as cv
import matplotlib.pyplot as plt
from baselines.a2c.utils import fc
import json
from progress.bar import Bar

MODEL_NAME = './checkpoints/mobilenet_v1_1.0_224'

img_size=224
factor=1.0
num_classes=1001
is_training=False
weight_decay = 0.0
freeze_mobilenet=False
tf.reset_default_graph()

inp = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 3),name="input")
labels=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
with tf.contrib.slim.arg_scope(arg_scope):
    logits, _ = mobilenet_v1.mobilenet_v1(inp,num_classes=num_classes,is_training=is_training,depth_multiplier=factor)

#predictions = tf.contrib.layers.softmax(logits)
#output = tf.identity(predictions, name='output')
features = tf.get_default_graph().get_tensor_by_name("MobilenetV1/Logits/AvgPool_1a/AvgPool:0")
sess = tf.Session()

if freeze_mobilenet:
    rest_var = tf.contrib.slim.get_variables_to_restore()
    var_dict={}
    for var in rest_var:
        noscope_name=var.name.replace(':0','')
        var_dict[noscope_name]=var  
    sess.run(tf.global_variables_initializer())
    saver_mobilenet = tf.train.Saver(var_dict)
    saver_mobilenet.restore(sess, MODEL_NAME+'.ckpt')

    #build detector
    features=tf.squeeze(features,[1,2])
    frozen_feat=tf.stop_gradient(features)
    with tf.variable_scope('Finetune/fc1'):
        h1 = fc(frozen_feat, 'fc', 256)
        h1 = tf.nn.tanh(h1)
    with tf.variable_scope('Finetune/fc2'):
        h2 = fc(h1, 'fc', 34)
        detection_logits = tf.identity(h2,name="detection_logits")
        loss=tf.losses.sigmoid_cross_entropy(labels,detection_logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
    new_variables = set(tf.global_variables()) - set()
    new_variables=[var for var in new_variables if 'Mobilenet' not in var.name]
    saver_finetuned=tf.train.Saver(new_variables)
    sess.run(tf.variables_initializer(new_variables))
else:
    #build detector
    features=tf.squeeze(features,[1,2])
    with tf.variable_scope('Finetune/fc1'):
        h2 = fc(features, 'fc', 34)
        detection_logits = tf.identity(h2,name="detection_logits")
        loss=tf.losses.sigmoid_cross_entropy(labels,detection_logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)

    sess.run(tf.global_variables_initializer())
    rest_var = tf.contrib.slim.get_variables_to_restore()
    var_dict={}
    for var in rest_var:
        if "Mobilenet" in var.name and "Finetune" not in var.name:
            noscope_name=var.name.replace(':0','')
            var_dict[noscope_name]=var  
    
    saver_mobilenet = tf.train.Saver(var_dict)
    saver_mobilenet.restore(sess, MODEL_NAME+'.ckpt')

    new_variables = set(tf.global_variables()) - set()
    new_variables=[var for var in new_variables if 'Mobilenet' not in var.name]
    saver_finetuned=tf.train.Saver()
    sess.run(tf.variables_initializer(new_variables))


#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

#get data and labels
INSTANCE_ID_MAP={
    "background" : 0,
    "advil_liqui_gels" : 1,
    "aunt_jemima_original_syrup" : 2,
    "bumblebee_albacore" : 3,
    "cholula_chipotle_hot_sauce" : 4,
    "coca_cola_glass_bottle" : 5,
    "crest_complete_minty_fresh" : 6,
    "crystal_hot_sauce" : 7,
    "expo_marker_red" : 8,
    "hersheys_bar" : 9,
    "honey_bunches_of_oats_honey_roasted" : 10,
    "honey_bunches_of_oats_with_almonds" : 11,
    "hunts_sauce" : 12,
    "listerine_green" : 13,
    "mahatma_rice" : 14,
    "nature_valley_granola_thins_dark_chocolate" : 15,
    "nutrigrain_harvest_blueberry_bliss" : 16,
    "pepto_bismol" : 17,
    "pringles_bbq" : 18,
    "progresso_new_england_clam_chowder" : 19,
    "quaker_chewy_low_fat_chocolate_chunk" : 20,
    "red_bull" : 21,
    "softsoap_clear" : 22,
    "softsoap_gold" : 23,
    "softsoap_white" : 24,
    "spongebob_squarepants_fruit_snaks" : 25,
    "tapatio_hot_sauce" : 26,
    "vo5_tea_therapy_healthful_green_tea_smoothing_shampoo" : 27,
    "nature_valley_sweet_and_salty_nut_almond" : 28,
    "nature_valley_sweet_and_salty_nut_cashew" : 29,
    "nature_valley_sweet_and_salty_nut_peanut" : 30,
    "nature_valley_sweet_and_salty_nut_roasted_mix_nut" : 31,
    "paper_plate" : 32,
    "red_cup" : 33
}

HOME_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'ActiveVisionDataset_downsampled/')
training_scenes=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1"]
#test_scenes=["Home_001_1","Home_014_1","Home_014_2"]
max_box_areas_file=open(os.path.join(HOME_DIR,"max_box_areas.json"))
max_box_areas=json.load(max_box_areas_file)

all_data=[]
for scene in training_scenes:
    scene_path = os.path.join(HOME_DIR,scene)
    images_path = os.path.join(scene_path,'jpg_rgb')
    image_names = os.listdir(images_path)
    annotations = json.load(open(os.path.join(scene_path,'annotations.json')))
    for image in image_names:
        boxes=annotations[image]['bounding_boxes']
        label=np.zeros(34)
        for box in boxes:
            box_area=(box[2]-box[0])*(box[3]-box[1])
            id=box[4]
            score=float(box_area)/max_box_areas[scene][str(id)]
            #label[id]+=score
            label[id]=1
        #all_labels.append(np.clip(label,None,1.0))
        all_data.append([os.path.join(images_path,image),label])


#train
training_epochs=500
batch_size=32
save_interval=1
train_test_split=0.9
num_training=round(len(all_data)*train_test_split)
indices = np.random.permutation(len(all_data))
training_idx, test_idx = indices[:num_training], indices[num_training:]
training=[all_data[i] for i in training_idx]
test=[all_data[i] for i in test_idx]
logfile = open('./checkpoints/finetuning_training_log.txt','a')
for epoch in range(training_epochs):
    avg_loss = 0.0
    total_batch = int(len(training) / batch_size)
    all_idx=np.random.permutation(len(training))
    bar = Bar('Epoch ' + str(epoch), max=total_batch, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for i in range(total_batch):
        #grab a random batch
        batch_idx=all_idx[:batch_size]
        all_idx=all_idx[batch_size:]
        train_batch=[training[i] for i in batch_idx]
        input_data=np.zeros((batch_size,224,224,3))
        labels_batch=np.zeros((batch_size,34))
        for i,data in enumerate(train_batch):
            input_data[i,...]=cv.imread(data[0])
            labels_batch[i,:]=data[1]

        _, loss_value = sess.run([train_op, loss], feed_dict={inp: input_data, labels: labels_batch})
        avg_loss += loss_value / total_batch
        bar.next()

    if epoch % save_interval == 0:
        if freeze_mobilenet:
            saver_finetuned.save(sess, "./checkpoints/finetune" + "%04d" % epoch + ".ckpt")
        else:
            saver_finetuned.save(sess, "./checkpoints/finetune_all" + "%04d" % epoch + ".ckpt")

    avg_test_loss = 0.0
    total_test_batch = int(len(test) / batch_size)
    all_idx=np.random.permutation(len(test))
    for i in range(total_test_batch):
        #grab a random batch
        batch_idx=all_idx[:batch_size]
        all_idx=all_idx[batch_size:]
        test_batch=[test[i] for i in batch_idx]
        input_data=np.zeros((batch_size,224,224,3))
        labels_batch=np.zeros((batch_size,34))
        for i,data in enumerate(test_batch):
            input_data[i,...]=cv.imread(data[0])
            labels_batch[i,:]=data[1]

        loss_value = sess.run(loss, feed_dict={inp: input_data, labels: labels_batch})
        avg_test_loss += loss_value / total_test_batch

    print("\tloss=", "{:.9f}".format(avg_loss),"test loss=","{:.9f}".format(avg_test_loss))
    logfile.write('\nEpoch ' + str(epoch) + "loss=" + "{:.9f}".format(avg_loss) + "test loss=" + "{:.9f}".format(avg_test_loss) )

logfile.close()




# #for var in tf.trainable_variables():
# # print(var.name)

# [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

# #test things out with an image
# img=cv.imread("/media/david/HardDrive/Documents/ActiveVisionDataset/Home_001_1/jpg_rgb/000110000010101.jpg",flags=-1)
# #img=cv.imread("ActiveVisionDataset_downsampled/Home_001_1/jpg_rgb/000110000010101.png",flags=-1)
# #img=cv.imread("ActiveVisionDataset/Home_001_1/jpg_rgb/000110000010101.jpg",flags=-1)
# #img=cv.imread("cat.jpg",flags=-1)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)[:,:,0:3]
# #img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)[:,:,0:3]
# #img=img[0:1080,420:1500,:]
# print(img.shape)
# img=cv.resize(img,(img_size,img_size))
# x=200
# y=400
# #img=rgb_image[x:x+img_size,y:y+img_size,:]
# fig,ax = plt.subplots(2)
# plt.cla()
# ax[0].imshow(img)

# img=np.expand_dims(img,axis=0)

# out = sess.run(output, feed_dict={inp:img})
# out=np.squeeze(out)
# print(out, out.shape)
# print(np.min(out),np.max(out))

# ax[1].plot(np.arange(0,out.size,1),out)
# plt.draw()
# plt.show()
