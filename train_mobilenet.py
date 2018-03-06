import math
import os
import matplotlib.pyplot as plt
import json
import random
import cv2 as cv
import numpy as np

from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence
from keras.losses import binary_crossentropy

ALPHA = 0.25
IMAGE_SIZE = 224
EPOCHS = 5000
PATIENCE = 100
BATCH_SIZE=32


NUM_IMAGES=21756
MAX_X=1920
MAX_Y=1080
NUM_REGRESSION=3
NUM_CLASSES=34

class DataSequence(Sequence):

    # def __load_images(self, dataset):
    #     out = []
    #     for file_name in dataset:
    #         im = cv2.resize(cv2.imread(file_name), (self.IMAGE_SIZE, self.IMAGE_SIZE))
    #         out.append(im)

    #     return np.array(out)

    def __init__(self, validation=False, batch_size=32, feature_scaling=False):
        HOME_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'ActiveVisionDataset_downsampled/')
        if validation:
            all_scenes=["Home_001_1","Home_014_1","Home_014_2"]
        else:
            all_scenes=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1"]
            #all_scenes=["Home_003_2"]
        
        max_box_areas_file=open(os.path.join(HOME_DIR,"max_box_areas.json"))
        max_box_areas=json.load(max_box_areas_file)

        self.y = []
        self.x = []
        self.image_size = IMAGE_SIZE
        for scene in all_scenes:
            print('loading scene: ', scene)
            scene_path = os.path.join(HOME_DIR,scene)
            images_path = os.path.join(scene_path,'jpg_rgb')
            depth_path = os.path.join(scene_path,'high_res_depth')
            image_names = os.listdir(images_path)
            annotations = json.load(open(os.path.join(scene_path,'annotations.json')))
            for image in image_names:
                depth_name=image[:-5]+'3.png'
                boxes=annotations[image]['bounding_boxes']
                img=cv.imread(os.path.join(images_path,image))
                img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

                self.x.append(img)
                y_vec=np.zeros((NUM_CLASSES,4))
                for box in boxes:
                    box_area=(box[2]-box[0])*(box[3]-box[1])
                    id=box[4]
                    y_vec[id,0] = float(box_area)/max_box_areas[scene][str(id)]
                    y_vec[id,1] = ((box[2]-box[0])/2.0+box[0])/MAX_X
                    y_vec[id,2] = ((box[3]-box[1])/2.0+box[1])/MAX_Y
                    y_vec[id,3] = 1
                self.y.append(y_vec)

        indices = np.random.permutation(len(self.x))
        self.x=[self.x[i] for i in indices]
        self.y=[self.y[i] for i in indices]
        self.x=np.stack(self.x)
        self.y=np.stack(self.y)

        self.batch_size = batch_size
        self.feature_scaling = feature_scaling
        if self.feature_scaling:
            dataset = self.x
            broadcast_shape = [1, 1, 1]
            broadcast_shape[2] = dataset.shape[3]

            self.mean = np.mean(dataset, axis=(0, 1, 2))
            self.mean = np.reshape(self.mean, broadcast_shape)
            self.std = np.std(dataset, axis=(0, 1, 2))
            self.std = np.reshape(self.std, broadcast_shape) + K.epsilon()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # images = self.__load_images(batch_x).astype('float32')
        if self.feature_scaling:
            batch_x -= self.mean
            batch_x /= self.std
        return batch_x, batch_y


def create_model(size, alpha):
    model_net = MobileNet(input_shape=(size, size, 3), include_top=False, alpha=alpha)
    x = _depthwise_conv_block(model_net.layers[-1].output, 1024, alpha, 1, block_id=14)
    x = MaxPooling2D(pool_size=(4, 4))(x)    
    reg = Conv2D(NUM_REGRESSION*NUM_CLASSES, kernel_size=(1, 1), padding="same")(x)
    reg = Reshape((NUM_CLASSES,NUM_REGRESSION))(reg)

    clas = Conv2D(NUM_CLASSES, kernel_size=(1, 1), padding="same", activation='softmax')(x)
    clas = Reshape((NUM_CLASSES,1))(clas)

    out=concatenate([reg,clas])
    return Model(inputs=model_net.input, outputs=out)


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.mean(x)

def custom_loss(y_true, y_pred):
    labels=y_true[:,:,-1:]
    num_objs=K.sum(labels)
    labels_mask=K.concatenate([labels for i in range(NUM_REGRESSION)],axis=-1)
    masked_reg_items=y_pred[:,:,:NUM_REGRESSION]*labels_mask
    box_loss = smoothL1(y_true[:,:,:NUM_REGRESSION]*labels_mask, masked_reg_items)*num_objs
    obj_loss = K.mean(binary_crossentropy(y_true[:,:,NUM_REGRESSION:], y_pred[...,NUM_REGRESSION:]))
    return box_loss + 100*obj_loss

def train(model, epochs, image_size):
    train_datagen = DataSequence(validation=False,batch_size=BATCH_SIZE)
    validation_datagen = DataSequence(validation=True,batch_size=BATCH_SIZE)

    model.compile(loss=custom_loss, optimizer="adam", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("model-{val_acc:.2f}.h5", monitor="val_acc", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=int(NUM_IMAGES/BATCH_SIZE), epochs=epochs, validation_data=validation_datagen,
                        validation_steps=22, callbacks=[checkpoint, stop])


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, IMAGE_SIZE)


if __name__ == "__main__":
    main()
