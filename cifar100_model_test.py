import argparse
import json
import random
import sys

import GPUtil as GPUtil
import pymongo
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from tf_model import TF_Model
from config import Config
import getopt

"""
专门用来手工调试和测试模型用的脚本
"""


# import os
# print(os.environ)
# Save class_names and n_classes for later

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://modelmarket:Qw123456789@naibo.wang:8088/', connect=False)
    mydb = myclient['exps']
    tasks = mydb["tasks"]
    results = mydb["results"]

    preset_config = {
        "model_name": "regnety200mf",
        "dataset": "cifar100",
        "required_image_size": 224,
        "remark": "artificial_test",
        "split": ["train[:400]", "train[400:500]", "test[:200]"],  # 写一个方法给定split方案
        "batch_size": 16,
        "epochs": 20,
        "GPUID": 0,
        "pretrained_model_path": os.path.join("/home/naibo/xacc_share/models/tf-dev/feature-extractor/",
                                              "regnety200mf_feature_extractor_1"),
        "storage_path": "/home/naibo/xacc_share/trained_models/",
        "normalize": False,
        "display_image": True,
        "random_seed": 2022,
        "learning_rate": 0.01,
    }
    c = Config(preset_config)
    c.set_command_line(sys.argv[1:])
    print(c)

    setup_seed(c.random_seed)  # 设置随机数种子
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[c.GPUID], device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    output_info = preset_config.copy()

    output_info["DEVICE"] = GPUtil.getGPUs()[0].name + ":" + str(c.GPUID)

    # strategy = tf.distribute.MirroredStrategy(
    #     devices=["/physical_device:GPU:1", "/physical_device:GPU:2","/physical_device:GPU:3","/physical_device:GPU:4","/physical_device:GPU:5","/physical_device:GPU:6","/physical_device:GPU:7", ])
    # with strategy.scope():

    tf_model = TF_Model(model_name=c.model_name, pretrained_model_path=c.pretrained_model_path,
                        storage_path=c.storage_path, split=c.split, remark=c.remark, output_info=output_info)
    tf_model.preprocess_dataset(required_image_size=[c.required_image_size, c.required_image_size],
                                batch_size=c.batch_size, normalize=c.normalize, display_image=c.display_image)
    tf_model.transfer_learning(epochs=c.epochs, learning_rate=c.learning_rate)
    tf_model.fine_tune(epochs=c.epochs, learning_rate=c.learning_rate)
    tf_model.test_pretrained_model("transfer-learning", save_predictions=True, display_image=True)
    tf_model.test_pretrained_model("fine-tune", save_predictions=True, display_image=True)
    output_info = tf_model.end_experiment()
    results.insert_one(output_info.copy())
    for t in output_info["time_records"]:
        t["start"] = t["start"].strftime('%Y-%m-%d %H:%M:%S.%f')
        t["end"] = t["end"].strftime('%Y-%m-%d %H:%M:%S.%f')
    with open(output_info["model_storage_path"] + "/output_info.json", "w") as f:
        json.dump(output_info, f)
