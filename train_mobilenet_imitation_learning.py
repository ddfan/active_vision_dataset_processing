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
from keras.losses import binary_crossentropy, categorical_crossentropy

ALPHA = 0.25
IMAGE_SIZE = 224
EPOCHS =10000
PATIENCE = 100
BATCH_SIZE=64


NUM_IMAGES=21756
MAX_X=1920
MAX_Y=1080
NUM_ACTIONS=7
NUM_CLASSES=34


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

class DataSequence(Sequence):

    # def __load_images(self, dataset):
    #     out = []
    #     for file_name in dataset:
    #         im = cv2.resize(cv2.imread(file_name), (self.IMAGE_SIZE, self.IMAGE_SIZE))
    #         out.append(im)

    #     return np.array(out)

    def __init__(self, test=False, validation=False, batch_size=32, feature_scaling=False):
        HOME_DIR='/media/david/HardDrive/Documents/ActiveVisionDataset_downsampled/'
        if validation:
            all_scenes=["Home_001_1","Home_014_1","Home_014_2"]
            # all_scenes=["Home_014_2"]
        else:
            all_scenes=["Home_003_2", "Home_005_2", "Home_010_1", "Home_001_2", "Home_004_1", "Home_006_1", "Home_011_1", "Home_015_1", "Home_002_1", "Home_004_2", "Home_007_1", "Home_013_1", "Home_016_1", "Home_003_1", "Home_005_1", "Home_008_1"]
            # all_scenes=["Home_003_2"]

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
            instance_names_path = os.path.join(scene_path,'present_instance_names.txt')
            instance_file = open(instance_names_path)
            instance_names = instance_file.read().splitlines()

            for image in image_names:
                depth_name=image[:-5]+'3.png'
                boxes=annotations[image]['bounding_boxes']
                img=cv.imread(os.path.join(images_path,image))
                img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

                self.x.append(img)
                y_vec=np.zeros((NUM_CLASSES,NUM_ACTIONS))
                for target_name in instance_names:
                    id=INSTANCE_ID_MAP[target_name]
                    move=annotations[image]['expert'][str(id)]
                    y_vec[id,move] = 1
                self.y.append(y_vec)

        np.random.seed(0)
        indices = np.random.permutation(len(self.x))
        test_train_split_ratio=0.1
        split_idx=round(test_train_split_ratio*len(indices))

        if test:
            indices=indices[:split_idx]
        else:
            indices=indices[split_idx:]
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

    clas = Conv2D(NUM_CLASSES*NUM_ACTIONS, kernel_size=(1, 1), padding="same", activation='softmax')(x)
    clas = Reshape((NUM_CLASSES,NUM_ACTIONS))(clas)
    return Model(inputs=model_net.input, outputs=clas)


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.mean(x)

def custom_loss(y_true, y_pred):
    obj_loss = K.mean(K.sum(categorical_crossentropy(y_true, y_pred),axis=-1))
    return obj_loss

def train(model, epochs, image_size):
    train_datagen = DataSequence(test=False,batch_size=BATCH_SIZE)
    validation_datagen = DataSequence(test=True,batch_size=BATCH_SIZE)

    model.compile(loss=custom_loss, optimizer="adam", metrics=["categorical_accuracy"])
    checkpoint = ModelCheckpoint("checkpoints/model-{val_categorical_accuracy:.2f}.h5", monitor="val_categorical_accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_categorical_accuracy", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=int(NUM_IMAGES/BATCH_SIZE), epochs=epochs, validation_steps=22, callbacks=[checkpoint, stop],
        validation_data=validation_datagen)


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, IMAGE_SIZE)


if __name__ == "__main__":
    main()
