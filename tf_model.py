import datetime
import json
import os
import random
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from preprocessing import _TFImageHelper, TimeRecord
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger


##TODO 参数化配置，mongodb的config表 分配好每个模型的gpuID以及是否已经finetune等
##batchnorm导致的fit和evaluate不一致的问题 - 两种方式的validation accuracy一致 training acc 不一致，是否和训练数据集随机打乱有关？ - 哪怕是transfer learning只训练最后一层，fit和evaluate在训练集上的acc也不同，所以也许不是batchnorm的事情！ - 忽略此现象，应该就是正常的
##保存transfer learning和finetune模型到不同的文件夹
##统计训练和测试时间（各个训练集）开始时间 结束时间等
##保存中间结果输出
##保存模型结构，模型各个轮的loss和acc
##测试early stopping和random seed
##根据是否为0-1来决定是否做除法
##TODO 统计每个模型的大小以及P100是否撑得住，图片显示是否正常；模型的原始数据集是什么，原始要求格式，原始的训练和测试准确率
##TODO WEBSITE 可视化每次的实验结果


class TF_Model:
    def __init__(self,
                 model_name,
                 pretrained_model_path,
                 dataset="cifar100",
                 split: List = ["train[:300]", "train[400:500]", "test[:100]"],
                 remark: str = "whole", #文件夹名称备注，通常指数据集比例
                 storage_path: str="/home/naibo/xacc_share/trained_models/",
                 output_info: dict={}) -> None:
        self.time_records = TimeRecord()
        self.model_name = model_name
        self.dataset = dataset
        self.split = split
        model_path = os.path.join(storage_path, model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        time_start = self.time_records.start("whole_experiment")
        self.storage_path = os.path.join(model_path,
                                         dataset + "_" + remark + "_" + time_start)
        # 创建模型的文件夹和相关日志
        os.mkdir(self.storage_path)
        os.mkdir(self.storage_path + "/fine-tune")
        os.mkdir(self.storage_path + "/transfer-learning")
        logger.add(self.storage_path + "/output.log")
        self.pretrained_model_path = pretrained_model_path
        self.output_info = output_info
        self.output_info["model_storage_path"] = self.storage_path
        self.output_info["results"] = []

    # 结束模型训练后，保存一些信息到数据库
    def end_experiment(self):
        self.time_records.end("whole_experiment")
        print(self.time_records.get_logs())
        self.output_info["time_records"] = self.time_records.get_logs()
        return self.output_info

    def preprocess_dataset(self,
                           required_image_size: List = [224, 224],
                           batch_size: int = 16,
                           normalize: bool = True,  # False: 保持图片原样（0-255） True:图片像素压缩到0-1
                           display_image: bool = False,  # 是否展示图片（处理前和处理后）
                           ):
        AUTO = tf.data.AUTOTUNE
        ds, ds_info = tfds.load(name=self.dataset,
                                split=self.split,
                                shuffle_files=True,
                                as_supervised=True,
                                with_info=True)

        self.image_size = required_image_size[0]
        self.num_classes = ds_info.features["label"].num_classes
        self.label_names = ds_info.features["label"].names

        logger.info("Reshaping images to {}".format(required_image_size))
        ds_train, ds_validation, ds_test = ds[0], ds[1], ds[2]

        # 可视化原始数据集
        if display_image:
            self.visualize_samples(ds_train)
            self.visualize_samples(ds_validation)
            self.visualize_samples(ds_test)

        ds_train = ds_train.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1]), normalize  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )
        ds_validation = ds_validation.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1]), normalize  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )
        ds_test = ds_test.map(
            lambda x, y: (
                _TFImageHelper.central_crop_with_resize_3_channels(
                    x, (required_image_size[0], required_image_size[1]), normalize  # 对于cifar100 从32x32放大到了224x224
                ),
                y,
            ), num_parallel_calls=AUTO
        )

        # 可视化处理后的数据集
        if display_image:
            self.visualize_preprocessed_samples(ds_train)
            self.visualize_preprocessed_samples(ds_validation)
            self.visualize_preprocessed_samples(ds_test)

        # shuffle是有效果的，但之所以每次测试的training set的分布都一样是因为之前设置了随机数种子的固定！
        # 一次shuffle后training set就打乱一次，不再变化了
        ds_train = ds_train.shuffle(buffer_size=10000)
        self.ds_train = ds_train.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        # 注意，验证集和测试集也需要使用batch！
        self.ds_validation = ds_validation.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        self.ds_test = ds_test.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()

        # for batch in self.ds_train:
        #     print(batch[0].shape) # print input shape
        #     break

    def transfer_learning(self,
                          learning_rate: float = 0.01,
                          epochs: int = 20):
        csv_logger = CSVLogger(self.storage_path + "/transfer-learning/training_log.csv", append=True, separator=',')
        # build the transfer learning model
        m = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.image_size, self.image_size, 3)),
                hub.KerasLayer(self.pretrained_model_path, trainable=False),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    self.num_classes,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                ),
            ]
        )
        m.build(input_shape=(None, self.image_size, self.image_size, 3))
        # base_model = tf.keras.models.load_model(self.pretrained_model_path)
        # base_model.trainable = False
        # # for layer in base_model.layers:
        # #     layer.trainable = True if isinstance(layer, tf.keras.layers.BatchNormalization) else False
        # # build the fine-tune model
        # # inputs = tf.keras.layers.InputLayer(input_shape=(self.image_size, self.image_size, 3))
        # inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        # x = base_model(inputs, training=False)  # 这里的training只能设置为true
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # outputs = tf.keras.layers.Dense(
        #     self.num_classes,
        #     activation="softmax",
        #     kernel_regularizer=tf.keras.regularizers.l2(0.0),
        # )(x)
        # m = tf.keras.Model(inputs, outputs)
        with open(os.path.join(self.storage_path, 'transfer-learning/model_summary.txt'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            m.summary(print_fn=lambda x: fh.write(x + '\n'))
        print(m.summary())

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
            self.ds_test
        )
        self.output_info["results"].append({"index":"testSet_before_transfer_learning","loss":eval_loss,"accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of test set of transfer learning (before): {}, {}".format(eval_loss, eval_acc))

        eval_loss, eval_acc = m.evaluate(
            self.ds_validation
        )
        self.output_info["results"].append({"index": "validationSet_before_transfer_learning", "loss": eval_loss, "accuracy": eval_acc})
        logger.info(
            "eval_loss, eval_acc of validation set of transfer learning (before): {}, {}".format(eval_loss, eval_acc))

        eval_loss, eval_acc = m.evaluate(
            self.ds_train
        )
        self.output_info["results"].append(
            {"index": "trainingSet_before_transfer_learning", "loss": eval_loss, "accuracy": eval_acc})
        logger.info(
            "eval_loss, eval_acc of training set of transfer learning (before): {}, {}\n".format(eval_loss, eval_acc))

        self.time_records.start("transfer_learning_fit")
        m.fit(
            self.ds_train,
            epochs=epochs,
            verbose=1,
            validation_data=self.ds_validation,
            callbacks=[csv_logger]
        )
        self.time_records.end("transfer_learning_fit")

        # for i in range(3):
        eval_loss, eval_acc = m.evaluate(
            self.ds_test,
            callbacks=[csv_logger]
        )
        self.output_info["results"].append(
            {"index": "testSet_after_transfer_learning", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of test set of transfer learning (after): {}, {}".format(eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            self.ds_validation
        )
        self.output_info["results"].append(
            {"index": "validationSet_after_transfer_learning", "loss": eval_loss, "accuracy": eval_acc})
        logger.info(
            "eval_loss, eval_acc of validation set of transfer learning (after): {}, {}".format(eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            self.ds_train
        )
        self.output_info["results"].append(
            {"index": "trainingSet_after_transfer_learning", "loss": eval_loss, "accuracy": eval_acc})
        logger.info(
            "eval_loss, eval_acc of training set of transfer learning (after): {}, {}\n".format(eval_loss, eval_acc))

        model_path = os.path.join(self.storage_path, "transfer-learning")
        self.save_predictions(m, model_path)

        logger.info("Saving to {}\n".format(model_path))
        m.save(model_path)
        return model_path

    def fine_tune(
            self,
            learning_rate: float = 0.01,
            epochs: int = 20,
    ):
        csv_logger = CSVLogger(self.storage_path + "/fine-tune/training_log.csv", append=True, separator=',')

        # base_model = tf.keras.models.load_model(self.pretrained_model_path)
        # base_model.trainable = True
        # for layer in base_model.layers:
        #     layer.trainable = False if isinstance(layer, tf.keras.layers.BatchNormalization) else True
        # # build the fine-tune model
        # inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        # x = base_model(inputs, training=True)  # 这里的training只能设置为true
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # outputs = tf.keras.layers.Dense(
        #     self.num_classes,
        #     activation="softmax",
        #     kernel_regularizer=tf.keras.regularizers.l2(0.0),
        # )(x)
        # m = tf.keras.Model(inputs, outputs)
        # base_model = hub.KerasLayer(self.pretrained_model_path, trainable=True),
        m = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.image_size, self.image_size, 3)),
                # base_model,
                # tf.keras.models.load_model(self.pretrained_model_path), # 效果与hub.KerasLayer相同，包括trainable=True的配置结果也是一样的
                hub.KerasLayer(self.pretrained_model_path, trainable=True),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    self.num_classes,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                ),
            ]
        )
        m.build(input_shape=(None, self.image_size, self.image_size, 3))

        with open(os.path.join(self.storage_path, 'fine-tune/model_summary.txt'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            m.summary(print_fn=lambda x: fh.write(x + '\n'))
        print(m.summary())
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
            self.ds_test
        )
        self.output_info["results"].append(
            {"index": "testSet_before_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of test set of fine tune (before): {}, {}".format(eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            self.ds_validation
        )
        self.output_info["results"].append(
            {"index": "validationSet_before_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of validation set of fine tune (before): {}, {}".format(eval_loss, eval_acc))
        # for i in range(10):
        eval_loss, eval_acc = m.evaluate(
            self.ds_train
        )
        self.output_info["results"].append(
            {"index": "trainingSet_before_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of training set of fine tune (before): {}, {}\n".format(eval_loss, eval_acc))

        # for i in range(10):
        #     for (d,l) in self.ds_validation: # 每次执行都一样
        #         print(l)
        # print("")
        # for i in range(10):
        #     for (d,l) in self.ds_test: # 每次执行都一样
        #         print(l)
        # print("")
        # for i in range(10):
        #     # random seed固定时，每次执行都一样，不固定时每次不同，但每次单独的实验中，10次的打印结果都一样，不同的实验如果seed不同则不同
        #     for (d,l) in self.ds_train:
        #         print(l)
        # print("")

        self.time_records.start("fine_tune_fit")
        m.fit(
            self.ds_train,
            epochs=epochs,
            verbose=1,
            validation_data=self.ds_validation,
            callbacks=[csv_logger]
        )
        self.time_records.end("fine_tune_fit")
        # for i in range(10): # 对training set evaluate 10次结果也相同
        eval_loss, eval_acc = m.evaluate(
            self.ds_test
        )
        self.output_info["results"].append(
            {"index": "testSet_after_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of test set of fine tune (after): {}, {}".format(eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            self.ds_validation
        )
        self.output_info["results"].append(
            {"index": "validationSet_after_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of validation set of fine tune (after): {}, {}".format(eval_loss, eval_acc))
        eval_loss, eval_acc = m.evaluate(
            self.ds_train
        )
        self.output_info["results"].append(
            {"index": "trainingSet_after_fine_tune", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of training set of fine tune (after): {}, {}\n".format(eval_loss, eval_acc))

        model_path = os.path.join(self.storage_path, "fine-tune")
        self.save_predictions(m, model_path)

        logger.info("Saving to {}\n".format(model_path))
        m.save(model_path)
        return model_path

    def test_pretrained_model(self, model_path="", display_image=False, save_predictions=False):
        if model_path == "fine-tune":
            model_path = os.path.join(self.storage_path, "fine-tune")
        elif model_path == "transfer-learning":
            model_path = os.path.join(self.storage_path, "transfer-learning")
        test_model = tf.keras.models.load_model(model_path)
        self.time_records.start(model_path + "_model_evaluate_test")
        eval_loss, eval_acc = test_model.evaluate(
            self.ds_test
        )
        self.time_records.end(model_path + "_model_evaluate_test")
        self.output_info["results"].append(
            {"index": model_path + "_model_evaluate_test", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of test set of the loaded model: {}, {}".format(eval_loss, eval_acc))
        # print("eval_loss, eval_acc of test set:", eval_loss, eval_acc)
        self.time_records.start(model_path + "_model_evaluate_validation")
        eval_loss, eval_acc = test_model.evaluate(
            self.ds_validation
        )
        self.time_records.end(model_path + "_model_evaluate_validation")
        self.output_info["results"].append(
            {"index": model_path + "_model_evaluate_validation", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of validation set of the loaded model: {}, {}".format(eval_loss, eval_acc))
        # print("eval_loss, eval_acc of validation set:", eval_loss, eval_acc)

        self.time_records.start(model_path + "_model_evaluate_training")
        eval_loss, eval_acc = test_model.evaluate(
            self.ds_train
        )
        self.time_records.end(model_path + "_model_evaluate_training")
        self.output_info["results"].append(
            {"index": model_path + "_model_evaluate_training", "loss": eval_loss, "accuracy": eval_acc})
        logger.info("eval_loss, eval_acc of training set of the loaded model: {}, {}\n".format(eval_loss, eval_acc))
        # print("eval_loss, eval_acc of training set:", eval_loss, eval_acc)
        if save_predictions:
            self.save_predictions(test_model, model_path, file_prefix="test_")

        if display_image:
            for batch_samples in self.ds_test:
                results = test_model.predict(batch_samples[0])
                predictions = np.argmax(results, axis=1)  # the prediction labels
                print(predictions, batch_samples[1])
                self.visualize_test_samples(batch_samples, predictions)
                break
            for batch_samples in self.ds_validation:
                results = test_model.predict(batch_samples[0])
                predictions = np.argmax(results, axis=1)  # the prediction labels
                print(predictions, batch_samples[1])
                self.visualize_test_samples(batch_samples, predictions)
                break
            for batch_samples in self.ds_train:
                results = test_model.predict(batch_samples[0])
                predictions = np.argmax(results, axis=1)  # the prediction labels
                print(predictions, batch_samples[1])
                self.visualize_test_samples(batch_samples, predictions)
                break
        return test_model

    def save_predictions(self, test_model, model_path, file_prefix=""):
        # 保存模型所有的输出
        x = {"labels": [], "predictions": []}
        for batch_samples in self.ds_test:
            results = test_model.predict(batch_samples[0])
            predictions = np.argmax(results, axis=1)  # the prediction labels
            x["predictions"].extend(predictions)
            x["labels"].extend(batch_samples[1].numpy())
        with open(os.path.join(model_path, file_prefix+"test_results.json"), 'w') as f:
            for i in range(len(x["labels"])):
                x["labels"][i] = str(x["labels"][i])
                x["predictions"][i] = str(x["predictions"][i])
            json.dump(x, f)

        x = {"labels": [], "predictions": []}
        for batch_samples in self.ds_validation:
            results = test_model.predict(batch_samples[0])
            predictions = np.argmax(results, axis=1)  # the prediction labels
            x["predictions"].extend(predictions)
            x["labels"].extend(batch_samples[1].numpy())
        with open(os.path.join(model_path, file_prefix+"validation_results.json"), 'w') as f:
            for i in range(len(x["labels"])):
                x["labels"][i] = str(x["labels"][i])
                x["predictions"][i] = str(x["predictions"][i])
            json.dump(x, f)

        x = {"labels": [], "predictions": []}
        for batch_samples in self.ds_train:
            results = test_model.predict(batch_samples[0])
            predictions = np.argmax(results, axis=1)  # the prediction labels
            x["predictions"].extend(predictions)
            x["labels"].extend(batch_samples[1].numpy())
        with open(os.path.join(model_path, file_prefix+"train_results.json"), 'w') as f:
            for i in range(len(x["labels"])):
                x["labels"][i] = str(x["labels"][i])
                x["predictions"][i] = str(x["predictions"][i])
            json.dump(x, f)

    def visualize_samples(self, ds):
        f, plots = plt.subplots(3, 3, figsize=(10, 10))
        images = []
        labels = []

        for sample in ds:
            if len(labels) > 9:
                break
            # imshow如果是float型数据，取值范围应在[0,1]；如果是int型数据，取值范围应在[0,255]。
            images.append(sample[0])
            labels.append(self.label_names[sample[1].numpy()])

        for i in range(3):
            for j in range(3):
                plots[i, j].imshow(images[i * 3 + j])
                plots[i, j].set_title(labels[i * 3 + j])
                plots[i, j].axis('off')

        plt.show()

    def visualize_preprocessed_samples(self, ds):
        f, plots = plt.subplots(3, 3, figsize=(10, 10))

        images = []
        labels = []

        for sample in ds:
            if len(labels) > 9:
                logger.info("preprocessed_sample:{}".format(sample))
                break
            # imshow显示浮点数的时候，只支持0～1之间的浮点数显示，超过1就认为是白色，所以在没有对值域做rescale的时候，中间的浮点数Mat显示只能是白色
            images.append(sample[0] / 255.0)  # 从0-1还原成0-255
            # images.append(sample[0])
            labels.append(self.label_names[sample[1].numpy()])

        for i in range(3):
            for j in range(3):
                plots[i, j].imshow(images[i * 3 + j])
                plots[i, j].set_title(labels[i * 3 + j])
                plots[i, j].axis('off')

        plt.show()

    def visualize_test_samples(self, batch_samples, predictions):
        f, plots = plt.subplots(3, 3, figsize=(10, 10))

        images = []
        labels = []
        for i in range(len(batch_samples[0])):
            sample = batch_samples[0][i]
            label = batch_samples[1][i]
            pred = predictions[i]
            if i > 9:
                break
            images.append(sample / 255.0)  # 从0-1还原成0-255
            labels.append(self.label_names[label.numpy()] + "-" + self.label_names[pred])

        for i in range(3):
            for j in range(3):
                plots[i, j].imshow(images[i * 3 + j])
                plots[i, j].set_title(labels[i * 3 + j])
                plots[i, j].axis('off')

        plt.show()
