import json
import os
import numpy as np
from functools import reduce


## tf.model.evaluate的结果和自己predict之后再比较的结果相同

def calculate_accuracy(file):
    with open(file, 'r') as f:
        x = json.load(f)
    for i in range(len(x["labels"])):
        x["labels"][i] = int(x["labels"][i])
        x["predictions"][i] = int(x["predictions"][i])
    # print(preds_tl)
    r = np.equal(x["labels"], x["predictions"])
    # print(r.sum(), len(r), "Acc of TL", r.sum() / len(r))
    return r.sum() / len(r)


def check_accuracy(info):
    if isinstance(info,str): # 如果传的是字符串，路径即此字符串，否则路径为info中保存的信息
        pretrained_model_path = info
        with open(pretrained_model_path + "/output_info.json", 'r') as f:
            info = json.load(f)
    else:
        pretrained_model_path = info["model_storage_path"]

    f = open(pretrained_model_path + "/check_accuracy.log", 'wt')
    acc = calculate_accuracy(pretrained_model_path + "/transfer-learning/test_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/transfer-learning/test_test_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "testSet_after_transfer_learning" else pre,
                      info["results"], None)
    acc_dict2 = reduce(lambda pre, cur: cur if cur[
                                                   'index'] == pretrained_model_path + "/transfer-learning_model_evaluate_test" else pre,
                       info["results"], None)
    # 后面三个判断：
    # 1. 直接训练出模型的准确率和保存后的模型的准确率是否相同
    # 2. 输出到文件的predictions计算出的准确率和output_info.json中记录的准确率是否相同
    # 3. output_info.json中记录的直接准确率和保存模型的准确率是否相同
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)
    acc = calculate_accuracy(pretrained_model_path + "/transfer-learning/validation_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/transfer-learning/test_validation_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "validationSet_after_transfer_learning" else pre,
                      info["results"], None)
    acc_dict2 = reduce(lambda pre, cur: cur if cur[
                                                   'index'] == pretrained_model_path + "/transfer-learning_model_evaluate_validation" else pre,
                       info["results"], None)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)

    acc = calculate_accuracy(pretrained_model_path + "/transfer-learning/train_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/transfer-learning/test_train_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "trainingSet_after_transfer_learning" else pre,
                      info["results"], None)
    acc_dict2 = reduce(lambda pre, cur: cur if cur[
                                                   'index'] == pretrained_model_path + "/transfer-learning_model_evaluate_training" else pre,
                       info["results"], None)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)

    acc = calculate_accuracy(pretrained_model_path + "/fine-tune/test_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/fine-tune/test_test_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "testSet_after_fine_tune" else pre,
                      info["results"], None)
    acc_dict2 = reduce(lambda pre, cur: cur if cur[
                                                   'index'] == pretrained_model_path + "/fine-tune_model_evaluate_test" else pre,
                       info["results"], None)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)

    acc = calculate_accuracy(pretrained_model_path + "/fine-tune/validation_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/fine-tune/test_validation_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "validationSet_after_fine_tune" else pre,
                      info["results"], None)
    acc_dict2 = reduce(lambda pre, cur: cur if cur[
                                                   'index'] == pretrained_model_path + "/fine-tune_model_evaluate_validation" else pre,
                       info["results"], None)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)

    acc = calculate_accuracy(pretrained_model_path + "/fine-tune/train_results.json")
    acc2 = calculate_accuracy(pretrained_model_path + "/fine-tune/test_train_results.json")
    acc_dict = reduce(lambda pre, cur: cur if cur['index'] == "trainingSet_after_fine_tune" else pre,
                      info["results"], None)
    acc_dict2 = reduce(
        lambda pre, cur: cur if cur['index'] == pretrained_model_path + "/fine-tune_model_evaluate_training" else pre,
        info["results"], None)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001)
    print(acc_dict["accuracy"], acc, acc2, acc_dict2["accuracy"], acc == acc2, (acc_dict["accuracy"] - acc) < 0.001,
          (acc_dict2["accuracy"] - acc) < 0.001, file=f)


if __name__ == '__main__':
    check_accuracy(
        "/home/naibo/xacc_share/trained_models/imagenet_mobilenet_v2_100_224/cifar100_artificial_test_2022_05_11_00_07_40_670267")
