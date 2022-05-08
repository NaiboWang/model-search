import argparse
import json
import random
import sys
from bson import ObjectId
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
1. 一个GPU可以同时跑多个程序/模型，且互相不影响（实验结果仍然一致），只要有显存
2. 可以使用MirroredStrategy使模型在多个GPU上同时跑，同时可以开多程序
3. 使用set_visible_devices来指定GPU
4. RANDOM_SEED固定后，每次的实验结果理论上应该一模一样，实际上不同（初始训练结果相同，后来不同）
5. Epoch设置为30比较合理
6. model.fit和model.evaluate在finetune的时候对训练集的结果不同，而transfer learning的时候是一样的，原因是finetune动了原始模型的一些只有训练过程中才会用到的层，如BatchNormalization的moving_mean和moving_variance不可finetune，Dropout，LayerNormalization等，所以导致不同；而transfer learning由于冻结了base layer所以结果完全相同
"""


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
        "GPUTYPE": 0,
        "GPUID": 0,
    }
    cg = Config(preset_config) # global configuration
    cg.set_command_line(sys.argv[1:]) # 命令行参数必须在preset_config中有定义！

    while True: # 一直运行任务直到没有任务可以执行
        undo_tasks = list(tasks.find({"status": "init", 'GPUTYPE': {'$lte': cg.GPUTYPE}})) # lte 小于等于
        if len(undo_tasks) == 0:
            break
        task = undo_tasks[0]
        c = Config(task)
        c.pretrained_model_path = "/home/naibo/xacc_share/models/tf-dev/feature-extractor/" + c.pretrained_model_path
        print(c) # print configurations
        tasks.update_one({"_id":task["_id"]}, {"$set": {"status": "running"}})

        setup_seed(c.random_seed)  # 设置随机数种子
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.set_visible_devices(devices=gpus[cg.GPUID], device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        output_info = preset_config.copy()

        output_info["DEVICE"] = GPUtil.getGPUs()[0].name + ":" + str(cg.GPUID)

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
        tasks.update_one({"_id": task["_id"]}, {"$set": {"status": "done"}})

