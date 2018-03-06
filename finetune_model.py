import os
import tensorflow as tf
import numpy as np
import mobilenet_v1
import cv2 as cv
import matplotlib.pyplot as plt
from baselines.a2c.utils import fc
from baselines.a2c.policies import nature_cnn
import json
from progress.bar import Bar
import random
import math

MODEL_NAME = './checkpoints/mobilenet_v1_1.0_224'
FINETUNE_NAME = './checkpoints/fullcnn1104'
RESUME_TRAINING_FROM=1104
TEST_NETWORK=False

img_size=224
factor=1.0
num_classes=1001
num_features=512
is_training=False
weight_decay = 0.0
network_type="fullcnn"
max_x=1920
max_y=1080
dropout_rate=0.9
training_epochs=10000
batch_size=64
save_interval=1
learning_rate=0.00001
train_test_split=0.9
tf.reset_default_graph()
sess = tf.Session()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def logit(x):
    return -tf.log(tf.reciprocal(x+1.0e-8)-(1.0-1.0e-7))

if network_type is "mobilenet_frozen":
    inp = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 3),name="input")
    labels_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    with tf.contrib.slim.arg_scope(arg_scope):
        logits, _ = mobilenet_v1.mobilenet_v1(inp,num_classes=num_classes,is_training=is_training,depth_multiplier=factor)
    features = tf.get_default_graph().get_tensor_by_name("MobilenetV1/Logits/AvgPool_1a/AvgPool:0")
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
        loss=tf.losses.sigmoid_cross_entropy(labels_ph,detection_logits)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
    new_variables = set(tf.global_variables()) - set()
    new_variables=[var for var in new_variables if 'Mobilenet' not in var.name]
    saver_finetuned=tf.train.Saver(new_variables)
    sess.run(tf.variables_initializer(new_variables))

if network_type is "mobilenet_splitoutput":
    inp = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 3),name="input")
    labels_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    with tf.contrib.slim.arg_scope(arg_scope):
        logits, _ = mobilenet_v1.mobilenet_v1(inp,num_classes=num_classes,is_training=is_training,depth_multiplier=factor)
    features = tf.get_default_graph().get_tensor_by_name("MobilenetV1/Logits/AvgPool_1a/AvgPool:0")

    with tf.variable_scope('Finetune'):
        depth_ph = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 1),name="depth")
        labels_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
        scores_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
        xpositions_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
        ypositions_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
        features=tf.squeeze(features,[1,2])
    #    frozen_feat=tf.stop_gradient(features)
        #features=tf.Print(features,[tf.reduce_min(features),tf.reduce_max(features)])

        h1 = fc(features, 'fc1', 34)
        h1 = tf.nn.sigmoid(h1)
        detection_logits = tf.identity(h1,name="detection_logits")
        loss1=tf.losses.mean_squared_error(labels_ph,detection_logits)

        h2 = fc(features, 'fc2', 34)
        h2 = tf.nn.sigmoid(h2)
        scores_out = tf.identity(h2,name="scores_out")
        scores_mask = tf.multiply(scores_out,labels_ph)
        loss2=tf.losses.mean_squared_error(scores_ph,scores_mask)

        h3 = fc(features, 'fc3', 34)
        h3 = tf.nn.sigmoid(h3)
        xpos_out = tf.identity(h3,name="xpos_out")
        xpos_mask = tf.multiply(xpos_out,labels_ph)
        loss3=tf.losses.mean_squared_error(xpositions_ph,xpos_mask)

        h4 = fc(features, 'fc4', 34)
        h4 = tf.nn.sigmoid(h4)
        ypos_out = tf.identity(h4,name="ypos_out")
        ypos_mask = tf.multiply(ypos_out,labels_ph)
        loss4=tf.losses.mean_squared_error(ypositions_ph,ypos_mask)

        weights = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in weights if 'b' not in v.name ]) * 0.004
        loss = loss1 + loss2 + loss3 + loss4 + lossL2

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
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

elif network_type is "fullcnn":
    inp = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 3),name="input")
    depth_ph = tf.placeholder(tf.float32,shape=(None, img_size, img_size, 1),name="depth")
    labels_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    scores_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    xpositions_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    ypositions_ph=tf.placeholder(tf.float32,shape=(None, 34),name="labels")
    dropout_rate_ph=tf.placeholder(tf.float32,shape=(),name="dropout_rate")
    cnninput=tf.concat([inp,depth_ph],axis=-1)
    conv_out=nature_cnn(cnninput)
    # conv_out=tf.nn.dropout(h0,keep_prob=0.8)
    h1a= fc(conv_out, 'fullcnn/detect/fc1', num_features)
    h1b = tf.nn.leaky_relu(h1a)
    h1b = tf.layers.dropout(h1b,rate=dropout_rate_ph)
    h1 = fc(h1b, 'fullcnn/detect/fc2', 34)
    #h1 = tf.nn.sigmoid(h1)
    detection_logits = tf.identity(h1,name="detection_logits")
    loss1=tf.losses.sigmoid_cross_entropy(labels_ph,detection_logits)

    h2 = fc(h1b, 'fullcnn/score/fc2', 34)
    # h2 = tf.nn.sigmoid(h2)
    scores_out = tf.identity(h2,name="scores_out")
    scores_mask = tf.multiply(scores_out,labels_ph)
    loss2=tf.losses.mean_squared_error(tf.multiply(logit(scores_ph),labels_ph),scores_mask)*tf.reduce_sum(labels_ph)
    #loss2=tf.losses.mean_squared_error(scores_ph,scores_mask)

    h3a= fc(conv_out, 'fullcnn/pos/fc1', num_features)
    h3b = tf.nn.leaky_relu(h3a)
    h3b = tf.layers.dropout(h3b,rate=dropout_rate_ph)
    h3 = fc(h3b, 'fullcnn/xpos/fc1', 34)
    # h3 = tf.nn.sigmoid(h3)
    xpos_out = tf.identity(h3,name="xpos_out")
    xpos_mask = tf.multiply(xpos_out,labels_ph)
    #loss3=tf.losses.mean_squared_error(tf.multiply(logit(xpositions_ph),labels_ph),xpos_mask)*tf.reduce_sum(labels_ph)
    loss3=tf.losses.mean_squared_error(xpositions_ph,xpos_mask)*tf.reduce_sum(labels_ph)*10

    h4 = fc(h3b, 'fullcnn/ypos/fc1', 34)
    # h4 = tf.nn.sigmoid(h4)
    ypos_out = tf.identity(h4,name="ypos_out")
    ypos_mask = tf.multiply(ypos_out,labels_ph)
    #loss4=tf.losses.mean_squared_error(tf.multiply(logit(ypositions_ph),labels_ph),ypos_mask)*tf.reduce_sum(labels_ph)
    loss4=tf.losses.mean_squared_error(ypositions_ph,ypos_mask)*tf.reduce_sum(labels_ph)*10

    weights = tf.trainable_variables() 
    loss = loss1 + loss2 + loss3*10 + loss4*10
    #loss = loss3 + loss4

    lossL2= loss + tf.add_n([ tf.nn.l2_loss(v) for v in weights if 'b' not in v.name ]) * 1.0e-6

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.0001)
    #train_op = optimizer.minimize(loss=loss)

    grads = tf.gradients(lossL2, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, 5.0) # gradient clipping
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver_finetuned=tf.train.Saver()
    # sess.run(tf.global_variables_initializer())

    if RESUME_TRAINING_FROM>0:
        #restore variables for finetuning
        rest_var = tf.contrib.slim.get_variables_to_restore()
        var_dict={}
        for var in rest_var:
            # if "fullcnn" not in var.name:
            #     continue
            noscope_name=var.name.replace(':0','')
            noscope_name=noscope_name.replace('model/','')
            var_dict[noscope_name]=var  
        saver_finetuned = tf.train.Saver(var_dict)
        saver_finetuned.restore(sess, './checkpoints/fullcnn'+"%04d" % RESUME_TRAINING_FROM+'.ckpt')



else:
    #build detector
    features=tf.squeeze(features,[1,2])
    with tf.variable_scope('Finetune/fc1'):
        h2 = fc(features, 'fc', 34)
        detection_logits = tf.identity(h2,name="detection_logits")
        loss=tf.losses.sigmoid_cross_entropy(labels_ph,detection_logits)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
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
#training_scenes=["Home_003_2"]
#test_scenes=["Home_001_1","Home_014_1","Home_014_2"]
max_box_areas_file=open(os.path.join(HOME_DIR,"max_box_areas.json"))
max_box_areas=json.load(max_box_areas_file)


if not TEST_NETWORK:
    print('loading data...')
    all_data=[]
    for scene in training_scenes:
        print('loading scene: ', scene)
        scene_path = os.path.join(HOME_DIR,scene)
        images_path = os.path.join(scene_path,'jpg_rgb')
        depth_path = os.path.join(scene_path,'high_res_depth')
        image_names = os.listdir(images_path)
        annotations = json.load(open(os.path.join(scene_path,'annotations.json')))
        for image in image_names:
            depth_name=image[:-5]+'3.png'
            boxes=annotations[image]['bounding_boxes']
            label=np.zeros(34)
            scores=np.zeros(34)
            xpositions=np.zeros(34)
            ypositions=np.zeros(34)
            input_dat=cv.imread(os.path.join(images_path,image))
            try:
                depth_dat=cv.imread(os.path.join(depth_path,depth_name))[:,:,0]
            except:
                depth_dat=cv.imread(os.path.join(images_path,image))[:,:,0]
            for box in boxes:
                box_area=(box[2]-box[0])*(box[3]-box[1])
                id=box[4]
                scores[id]=float(box_area)/max_box_areas[scene][str(id)]
                xpositions[id]=((box[2]-box[0])/2.0+box[0])/max_x
                ypositions[id]=((box[3]-box[1])/2.0+box[1])/max_y
                #label[id]+=score
                label[id]=1
            #all_labels.append(np.clip(label,None,1.0))
            # all_data.append([os.path.join(images_path,image),label,scores,xpositions,ypositions,os.path.join(depth_path,depth_name)])
            all_data.append([input_dat,label,scores,xpositions,ypositions,depth_dat])
    #train
    np.random.seed(100)
    num_training=round(len(all_data)*train_test_split)
    indices = np.random.permutation(len(all_data))
    training_idx, test_idx = indices[:num_training], indices[num_training:]
    training=[all_data[i] for i in training_idx]
    test=[all_data[i] for i in test_idx]
    logfile = open('./checkpoints/finetuning_training_log.txt','a')
    for epoch in range(training_epochs):
        print_epoch=RESUME_TRAINING_FROM+epoch
        avg_loss = np.zeros(5)
        total_batch = int(len(training) / batch_size)
        all_idx=np.random.permutation(len(training))
        bar = Bar('Epoch ' + str(print_epoch), max=total_batch, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        for i in range(total_batch):
            #grab a random batch
            batch_idx=all_idx[:batch_size]
            all_idx=all_idx[batch_size:]
            train_batch=[training[i] for i in batch_idx]
            input_data=np.zeros((batch_size,224,224,3))
            depth_data=np.zeros((batch_size,224,224,1))
            labels_batch=np.zeros((batch_size,34))
            scores_batch=np.zeros((batch_size,34))
            xpos_batch=np.zeros((batch_size,34))
            ypos_batch=np.zeros((batch_size,34))
            for i,data in enumerate(train_batch):
                # input_data[i,...]=cv.imread(data[0])
                # try:
                #     depth_data[i,:,:,0]=cv.imread(data[5])[:,:,0]
                # except:
                #     depth_data[i,:,:,0]=cv.imread(data[0])[:,:,0]
                input_data[i,...]=data[0]
                depth_data[i,:,:,0]=data[5]
                labels_batch[i,:]=data[1]
                scores_batch[i,:]=data[2]
                xpos_batch[i,:]=data[3]
                ypos_batch[i,:]=data[4]

            if network_type is "fullcnn" or network_type is "mobilenet_splitoutput":
                _, loss_value, loss1_val, loss2_val, loss3_val, loss4_val = sess.run([train_op, loss, loss1, loss2, loss3, loss4], feed_dict={inp: input_data,
                                                                labels_ph: labels_batch,
                                                                scores_ph: scores_batch,
                                                                xpositions_ph: xpos_batch,
                                                                ypositions_ph: ypos_batch,
                                                                depth_ph: depth_data,
                                                                dropout_rate_ph: dropout_rate})
            else:
                _, loss_value = sess.run([train_op, loss], feed_dict={inp: input_data,labels_ph: labels_batch})

            avg_loss[0] += loss_value / total_batch 
            avg_loss[1] += loss1_val / total_batch
            avg_loss[2] += loss2_val / total_batch
            avg_loss[3] += loss3_val / total_batch
            avg_loss[4] += loss4_val / total_batch
            bar.next()

        if epoch % save_interval == 0:
            if network_type is "mobilenet_frozen":
                saver_finetuned.save(sess, "./checkpoints/finetune" + "%05d" % print_epoch + ".ckpt")
            elif network_type is "mobilenet_splitoutput":
                saver_finetuned.save(sess, "./checkpoints/finetune_split" + "%05d" % print_epoch + ".ckpt")
            elif network_type is "fullcnn":
                saver_finetuned.save(sess, "./checkpoints/fullcnn" + "%05d" % print_epoch + ".ckpt")
            else:
                saver_finetuned.save(sess, "./checkpoints/finetune_all" + "%05d" % print_epoch + ".ckpt")

        avg_test_loss = 0.0
        total_test_batch = int(len(test) / batch_size)
        all_idx=np.random.permutation(len(test))
        for i in range(total_test_batch):
            #grab a random batch
            batch_idx=all_idx[:batch_size]
            all_idx=all_idx[batch_size:]
            test_batch=[test[i] for i in batch_idx]
            input_data=np.zeros((batch_size,224,224,3))
            depth_data=np.zeros((batch_size,224,224,1))
            labels_batch=np.zeros((batch_size,34))
            for i,data in enumerate(test_batch):
                # input_data[i,...]=cv.imread(data[0])
                # try:
                #     depth_data[i,:,:,0]=cv.imread(data[5])[:,:,0]
                # except:
                #     depth_data[i,:,:,0]=cv.imread(data[0])[:,:,0]
                input_data[i,...]=data[0]
                depth_data[i,:,:,0]=data[5]
                labels_batch[i,:]=data[1]
                scores_batch[i,:]=data[2]
                xpos_batch[i,:]=data[3]
                ypos_batch[i,:]=data[4]

            if network_type is "fullcnn" or network_type is "mobilenet_splitoutput":
                loss_value = sess.run(loss, feed_dict={inp: input_data,
                                                    labels_ph: labels_batch,
                                                    scores_ph: scores_batch,
                                                    xpositions_ph: xpos_batch,
                                                    ypositions_ph: ypos_batch,
                                                    depth_ph: depth_data,
                                                    dropout_rate_ph:1.0})


            else:
                loss_value = sess.run(loss, feed_dict={inp: input_data, labels_ph: labels_batch})

            avg_test_loss += loss_value / total_test_batch

        print('\nEpoch',str(print_epoch),"l=", "{:.3e}".format(avg_loss[0]),"l1=", "{:.3e}".format(avg_loss[1]),"l2=", "{:.3e}".format(avg_loss[2]),"l3=", "{:.3e}".format(avg_loss[3]),"l4=", "{:.3e}".format(avg_loss[4]),
            "test loss=","{:.3e}".format(avg_test_loss))
        logfile.write('\nEpoch ' + str(print_epoch) + "loss=" + "{:.9e}".format(avg_loss[0]) + "test loss=" + "{:.9e}".format(avg_test_loss) )

    logfile.close()


else:
    sess.run(tf.global_variables_initializer())

    #restore variables for finetuning
    rest_var = tf.contrib.slim.get_variables_to_restore()
    var_dict={}
    for var in rest_var:
        if "fullcnn" not in var.name:
            continue
        noscope_name=var.name.replace(':0','')
        noscope_name=noscope_name.replace('model/','')
        var_dict[noscope_name]=var  
    saver_finetuned = tf.train.Saver(var_dict)
    saver_finetuned.restore(sess, FINETUNE_NAME+'.ckpt')

    all_data=[]
    for scene in training_scenes:
        scene_path = os.path.join(HOME_DIR,scene)
        images_path = os.path.join(scene_path,'jpg_rgb')
        depth_path = os.path.join(scene_path,'high_res_depth')
        image_names = os.listdir(images_path)
        annotations = json.load(open(os.path.join(scene_path,'annotations.json')))
        for image in image_names:
            depth_name=image[:-5]+'3.png'
            boxes=annotations[image]['bounding_boxes']
            label=np.zeros(34)
            scores=np.zeros(34)
            xpositions=np.zeros(34)
            ypositions=np.zeros(34)
            # input_dat=cv.imread(os.path.join(images_path,image))
            # try:
            #     depth_dat=cv.imread(os.path.join(depth_path,depth_name))[:,:,0]
            # except:
            #     depth_dat=cv.imread(os.path.join(images_path,image))[:,:,0]
            for box in boxes:
                box_area=(box[2]-box[0])*(box[3]-box[1])
                id=box[4]
                scores[id]=float(box_area)/max_box_areas[scene][str(id)]
                xpositions[id]=((box[2]-box[0])/2.0+box[0])/max_x
                ypositions[id]=((box[3]-box[1])/2.0+box[1])/max_y
                #label[id]+=score
                label[id]=1
            #all_labels.append(np.clip(label,None,1.0))
            all_data.append([os.path.join(images_path,image),label,scores,xpositions,ypositions,os.path.join(depth_path,depth_name)])
            #all_data.append([input_dat,label,scores,xpositions,ypositions,depth_dat])

    np.random.seed(100)
    num_training=round(len(all_data)*train_test_split)
    indices = np.random.permutation(len(all_data))
    training_idx, test_idx = indices[:num_training], indices[num_training:]
    training=[all_data[i] for i in training_idx]

    while True:
        data=random.choice(training)
        #test things out with an image
        #img=cv.imread("/media/david/HardDrive/Documents/ActiveVisionDataset/Home_001_1/jpg_rgb/000110000010101.jpg",flags=-1)
        img=cv.imread(data[0])
        #img=cv.imread("ActiveVisionDataset/Home_001_1/jpg_rgb/000110000010101.jpg",flags=-1)
        #img=cv.imread("cat.jpg",flags=-1)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)[:,:,0:3]
        #img=img[0:1080,420:1500,:]
        img_inp=np.expand_dims(img,axis=0)
        depth_data=cv.imread(data[5])[:,:,0:1]
        depth_data=np.expand_dims(depth_data,axis=0)
        #img=cv.resize(img,(img_size,img_size))
        #x=200
        #y=400
        #img=rgb_image[x:x+img_size,y:y+img_size,:]
        
        detect,score,xpos,ypos = sess.run([detection_logits,scores_out,xpos_out,ypos_out], feed_dict={inp:img_inp,depth_ph: depth_data,dropout_rate_ph:dropout_rate})
        #print(detect,score,xpos,ypos)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.cla()
        plt.imshow(img)

        label=data[1]
        scores_true=data[2]
        xpos_true=data[3]
        ypos_true=data[4]
        for i,l in enumerate(label):
            if l:
                xtrue=xpos_true[i]*img_size
                ytrue=ypos_true[i]*img_size
                plt.text(xtrue+5, ytrue-5, '{:0.3f}'.format(scores_true[i]), fontsize=10, color='g')
                plt.plot(xtrue,ytrue,'go',markersize=10,markerfacecolor='None')

                xout=(xpos[0,i])*img_size
                yout=(ypos[0,i])*img_size
                plt.text(xout+5, yout-5, '{:0.3f}'.format(sigmoid(score[0,i])), fontsize=10, color='r')
                plt.plot(xout,yout,'ro',markersize=10,markerfacecolor='None')            

                plt.plot([xtrue, xout],[ytrue, yout],'b')
        plt.draw()
        plt.show()

