import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from preprocessing import _TFImageHelper
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log_cifar100.csv', append=True, separator=';')

label_dict = dict({
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "crab",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm",
})
logger.add("cifar100_1percent.log")

def visualize_samples(ds):
    f, plots = plt.subplots(3, 3, figsize=(10, 10))
    images = []
    labels = []

    for sample in ds:
        if len(labels) > 9:
            break
        # imshow如果是float型数据，取值范围应在[0,1]；如果是int型数据，取值范围应在[0,255]。
        images.append(sample[0])
        labels.append(label_dict[sample[1].numpy()])

    for i in range(3):
        for j in range(3):
            plots[i, j].imshow(images[i * 3 + j])
            plots[i, j].set_title(labels[i * 3 + j])
            plots[i, j].axis('off')

    plt.show()


def visualize_preprocessed_samples(ds):
    f, plots = plt.subplots(3, 3, figsize=(10, 10))

    images = []
    labels = []

    for sample in ds:
        if len(labels) > 9:
            logger.info("preprocessed_sample:{}".format(sample))
            break
        # imshow显示浮点数的时候，只支持0～1之间的浮点数显示，超过1就认为是白色，所以在没有对值域做rescale的时候，中间的浮点数Mat显示只能是白色
        #TODO 根据是否为0-1来决定是否做除法
        images.append(sample[0]/255.0)  # 从0-1还原成0-255
        labels.append(label_dict[sample[1].numpy()])

    for i in range(3):
        for j in range(3):
            plots[i, j].imshow(images[i * 3 + j])
            plots[i, j].set_title(labels[i * 3 + j])
            plots[i, j].axis('off')

    plt.show()

def visualize_test_samples(batch_samples, preds):
    f, plots = plt.subplots(3, 3, figsize=(10, 10))

    images = []
    labels = []
    for i in range(len(batch_samples[0])):
        sample = batch_samples[0][i]
        label = batch_samples[1][i]
        pred = preds[i]
        if i > 9:
            break
        images.append(sample/255.0)  # 从0-1还原成0-255
        labels.append(label_dict[label.numpy()] + "-" + label_dict[pred])

    for i in range(3):
        for j in range(3):
            plots[i, j].imshow(images[i * 3 + j])
            plots[i, j].set_title(labels[i * 3 + j])
            plots[i, j].axis('off')

    plt.show()

CROP_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 128
AUTO = tf.data.AUTOTUNE
def preprocess_train(images, labels):
    aug_images = tf.image.resize(images, (320, 320))
    aug_images = tf.image.random_flip_left_right(aug_images)
    aug_images = tf.image.random_crop(aug_images, size=(CROP_SIZE, CROP_SIZE, 3))

    one_hot_labels = tf.cast(labels, tf.uint8) # convert label type to uint8
    one_hot_labels = tf.one_hot(one_hot_labels, depth=NUM_CLASSES) # convert label to one hot
    return aug_images, labels

def preprocess_val(images, labels):
    aug_images = tf.image.resize(images, (256, 256))
    aug_images = tf.image.central_crop(aug_images, 224./256.) # The input should be 224 finally

    one_hot_labels = tf.cast(labels, tf.uint8)
    one_hot_labels = tf.one_hot(one_hot_labels, depth=NUM_CLASSES)
    return aug_images, labels

class Finetuner:
    def __init__(self) -> None:
        self.storage = "/home/naibo/xacc_share/models/finetuned/"

    def finetune(
            self,
            required_image_size: List = [224, 224],
            model_path: str = "/home/naibo/xacc_share/models/tf-dev/feature-extractor/regnety400mf_feature_extractor_1",
            dataset: str = "cifar100",
            split: List = ["train[:400]", "train[400:500]", "test"],
            num_labels: int = 100,
            learning_rate: float = 0.01,
            epochs: int = 20,
            batch_size: int = 16,
    ):

        AUTO = tf.data.AUTOTUNE
        # model = tf.keras.models.load_model(model_path)
        # model = hub.load(model)
        ds, ds_info = tfds.load(name=dataset,
                                split=split,
                                shuffle_files=True,
                                as_supervised=True,
                                with_info=True)
        ds_train, ds_validation, ds_test = ds[0], ds[1], ds[2]
        logger.info("Reshaping images to {}".format(required_image_size))
        # for sample in ds_train:
        #     plt.imshow(sample[0])
        #     plt.show()
        #     break
        # visualize_samples(ds_train)
        # visualize_samples(ds_validation)
        # visualize_samples(ds_test)

        # for x in ds_train:
        #     t = _TFImageHelper.central_crop_with_resize_3_channels(
        #         x, (required_image_size[0], required_image_size[1])  # 对于cifar100 从32x32放大到了224x224
        #     )

        ds_train = ds_train.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1])  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )
        ds_validation = ds_validation.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1])  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )
        ds_test = ds_test.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1])  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )

        # ds_train = ds_train.map(preprocess_train, num_parallel_calls=AUTO)
        # ds_validation = ds_validation.map(preprocess_val, num_parallel_calls=AUTO)
        # ds_test = ds_test.map(preprocess_val, num_parallel_calls=AUTO)

        visualize_preprocessed_samples(ds_train)
        # visualize_preprocessed_samples(ds_validation)
        # visualize_preprocessed_samples(ds_test)

        # for sample in ds_train:
        #     t2 = sample
        #     print(sample)
        #     plt.imshow(sample[0])
        #     plt.show()
        #     break
        image_size = required_image_size[0]
        num_classes = ds_info.features["label"].num_classes
        ds_train = ds_train.shuffle(buffer_size=10000)
        ds_train = ds_train.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        # 注意，验证集和测试集也需要使用batch！
        ds_validation = ds_validation.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        ds_test = ds_test.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        # for batch in ds_train:
        #     print(batch[0].shape) # print input shape
        #     break
        # now construct finetunning model

        ### build the transfer learning model
        m = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
                hub.KerasLayer(model_path, trainable=False),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                ),
            ]
        )
        m.build(input_shape=(None, image_size, image_size, 3))
        logger.info(m.summary())

        m.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9, nesterov=True
            ),
            # optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            # metrics=["accuracy", "tp", "fp", "tn", "fn", "precision", "recall"],
            metrics=["accuracy"]
        )
        eval_loss, eval_acc = m.evaluate(
            ds_test
        )
        logger.info("eval_loss, eval_acc of test set of transfer learning (before): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_validation
        )
        logger.info("eval_loss, eval_acc of validation set of transfer learning (before): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_train
        )
        logger.info("eval_loss, eval_acc of training set of transfer learning (before): {}, {}".format( eval_loss, eval_acc))

        m.fit(
            ds_train,
            epochs=epochs,
            verbose=1,
            validation_data=ds_validation,
            callbacks=[csv_logger]
        )
        eval_loss, eval_acc = m.evaluate(
            ds_test,
            callbacks=[csv_logger]
        )
        logger.info("eval_loss, eval_acc of test set of transfer learning (after): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_validation
        )
        logger.info("eval_loss, eval_acc of validation set of transfer learning (after): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_train
        )
        logger.info("eval_loss, eval_acc of training set of transfer learning (after): {}, {}".format( eval_loss, eval_acc))




        ### build the finetune model
        m = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
                hub.KerasLayer(model_path, trainable=True),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                ),
            ]
        )
        m.build(input_shape=(None, image_size, image_size, 3))
        logger.info(m.summary())
        m.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9, nesterov=True
            ),
            # optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            # metrics=["accuracy", "tp", "fp", "tn", "fn", "precision", "recall"],
            metrics=["accuracy"]
        )
        eval_loss, eval_acc = m.evaluate(
            ds_test
        )
        logger.info("eval_loss, eval_acc of test set of finetune (before): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_validation
        )
        logger.info("eval_loss, eval_acc of validation set of finetune (before): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_train
        )
        logger.info("eval_loss, eval_acc of training set of finetune (before): {}, {}".format( eval_loss, eval_acc))

        m.fit(
            ds_train,
            epochs=epochs,
            verbose=1,
            validation_data=ds_validation,
            callbacks=[csv_logger]
        )
        eval_loss, eval_acc = m.evaluate(
            ds_test
        )
        logger.info("eval_loss, eval_acc of test set of finetune (after): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_validation
        )
        logger.info("eval_loss, eval_acc of validation set of finetune (after): {}, {}".format( eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            ds_train
        )
        logger.info("eval_loss, eval_acc of training set of finetune (after): {}, {}".format( eval_loss, eval_acc))





        model_path = os.path.join(self.storage, "test")
        logger.info("Saving to ...{}".format(model_path))
        m.save(model_path)
        test_model = tf.keras.models.load_model("/home/naibo/exps/model-search/models/finetuned/test")

        eval_loss, eval_acc = test_model.evaluate(
            ds_test
        )
        print("eval_loss, eval_acc of test set:", eval_loss, eval_acc)
        eval_loss, eval_acc = test_model.evaluate(
            ds_validation
        )
        print("eval_loss, eval_acc of validation set:", eval_loss, eval_acc)
        eval_loss, eval_acc = test_model.evaluate(
            ds_train
        )
        print("eval_loss, eval_acc of training set:", eval_loss, eval_acc)

        for batch_samples in ds_test:
            results = test_model.predict(batch_samples[0])
            preds = np.argmax(results,axis=1) # the prediction labels
            print(preds, batch_samples[1])
            visualize_test_samples(batch_samples,preds)
            break
        for batch_samples in ds_validation:
            results = test_model.predict(batch_samples[0])
            preds = np.argmax(results, axis=1)  # the prediction labels
            print(preds, batch_samples[1])
            visualize_test_samples(batch_samples, preds)
            break
        for batch_samples in ds_train:
            results = test_model.predict(batch_samples[0])
            preds = np.argmax(results, axis=1)  # the prediction labels
            print(preds, batch_samples[1])
            visualize_test_samples(batch_samples, preds)
            break
        return model_path
