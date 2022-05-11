# -*- coding: utf-8 -*-
import random
import sys


from config import Config
from generate_task import insert
from tf_model import TF_Model


import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install

if __name__ == '__main__':
    print("TF version:", tf.__version__)
    print("Hub version:", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[1], device_type='GPU')
    preset_config = {
            "model_name": "regnety200mf",
            "dataset": "cifar100",
            "required_image_size": 224,
            "remark": "artificial_test",
            "split": ["train[:400]", "train[400:500]", "test[:200]"],  # 写一个方法给定split方案
            "batch_size": 16,
            "epochs": 10,
            "GPUTYPE":0,
            "GPUID": 2,
            "pretrained_model_path": os.path.join("/home/naibo/xacc_share/models/tf-dev/feature-extractor/",
                                                  "regnety200mf_feature_extractor_1"),
            "storage_path": "/home/naibo/xacc_share/trained_models/",
            "normalize": False,
            "display_image": True,
            "random_seed": 2022,
            "learning_rate": 0.01,
            "original_dataset":"imagenet1k",
            "original_accuracy":0.01,
            "status":"done",
        }
    latest_config = insert(insert=False,preset_config=preset_config) # 根据model_info.xls的最后一条模型配置生成新模型训练参数
    c = Config(latest_config)
    c.set_command_line(sys.argv[1:])
    c.pretrained_model_path = "/home/naibo/xacc_share/models/tf-dev/feature-extractor/" + c.pretrained_model_path
    print(c)
    setup_seed(c.random_seed)  # 设置随机数种子
    output_info = c.get_config()

    tf_model = TF_Model(model_name=c.model_name, pretrained_model_path=c.pretrained_model_path,
                        storage_path=c.storage_path, split=c.split, remark=c.remark, output_info=output_info)
    train_ds, val_ds, _ = tf_model.preprocess_dataset(required_image_size=[c.required_image_size, c.required_image_size],
                                batch_size=c.batch_size, normalize=c.normalize, display_image=False)

    BATCH_SIZE = 16#@param {type:"integer"}

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224,224,3)),
        hub.KerasLayer("/home/naibo/xacc_share/models/tf-dev/feature-extractor/imagenet_mobilenet_v2_100_224_feature_vector_5", trainable=True),
        tf.keras.layers.Dense(
            100,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.0),
        ),
    ])
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    """## Training the model"""

    model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])

    # model.evaluate(train_ds)

    # for sample in train_ds:
    #     print(sample,sample[0].shape,sample[1].shape)
    #     break


    hist = model.fit(
        train_ds,
        epochs=20,
        verbose=1,
        validation_data=val_ds)