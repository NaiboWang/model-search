import os
from functools import reduce
import pandas as pd
import pymongo
import csv

def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return round(size/1024/1024,1)

if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://modelmarket:Qw123456789@naibo.wang:8088/', connect=False)
    mydb = myclient['exps']
    results = mydb["results"]
    exp_results = list(results.find())  # lte 小于等于
    row_title = []
    for key in exp_results[0]:
        row_title.append(key)
    row_title.remove("results")
    row_title.remove("time_records")
    row_title.remove("check_accuracy_log")
    row_title.remove("normalize")
    row_title.remove("display_image")
    row_title.remove("storage_path")
    row_title.remove("pretrained_model_path")
    row_title.remove("model_storage_path")
    row_title.remove("status")
    row_title.remove("_id")
    print(row_title)
    row_title2 = []
    # row_title2.append("testSet_before_transfer_learning")
    # row_title2.append("validationSet_before_transfer_learning")
    # row_title2.append("trainingSet_before_transfer_learning")
    row_title2.append("testSet_after_transfer_learning")
    row_title2.append("validationSet_after_transfer_learning")
    row_title2.append("trainingSet_after_transfer_learning")
    # row_title2.append("testSet_before_fine_tune")
    # row_title2.append("validationSet_before_fine_tune")
    # row_title2.append("trainingSet_before_fine_tune")
    row_title2.append("testSet_after_fine_tune")
    row_title2.append("validationSet_after_fine_tune")
    row_title2.append("trainingSet_after_fine_tune")

    row_titles = []
    for key in row_title:
        row_titles.append(key)
    for key in row_title2:
        row_titles.append(key)
    row_titles.append("model_size")
    row_titles.append("test_time(s)")
    row_titles.append("tl_fit_time(min)")
    row_titles.append("ft_fit_time(min)")
    rows = []
    for result in exp_results:
        row = []
        for key in row_title:
            row.append(str(result[key]))
        for key in row_title2:
            accuracy = reduce(lambda pre, cur: cur if cur['index'] == key else pre, result["results"],
                                 None)["accuracy"]  # 对象数组查找
            row.append(str(accuracy))
        path = result["model_storage_path"] + "/fine-tune"
        model_size = get_dir_size(path)
        row.append(model_size)
        time_dict = reduce(lambda pre, cur: cur if cur['name'] == result[
            "model_storage_path"] + "/fine-tune_model_evaluate_training" else pre, result["time_records"], None)
        test_time = time_dict["end"] - time_dict["start"]
        # print("Test time:", test_time.total_seconds(), "s")
        row.append(str(test_time.total_seconds()))
        time_dict = reduce(lambda pre, cur: cur if cur['name'] == "transfer_learning_fit" else pre,
                           result["time_records"], None)
        transfer_learning_time = time_dict["end"] - time_dict["start"]
        # print("transfer learning training time:", transfer_learning_time.total_seconds() // 60, "min")
        row.append(str(transfer_learning_time.total_seconds()//60))
        time_dict = reduce(lambda pre, cur: cur if cur['name'] == "fine_tune_fit" else pre, result["time_records"],
                           None)
        fine_tune_time = time_dict["end"] - time_dict["start"]
        # print("fine tune training time:", fine_tune_time.total_seconds() // 60, "min")
        row.append(str(fine_tune_time.total_seconds()//60))
        print(row)

        rows.append(row)
    with open("result_format.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_titles)
        for r in rows:
            writer.writerow(r)

    data_frame = pd.read_csv("result_format.csv")
    rank_test_fine_tune = data_frame.sort_values("testSet_after_fine_tune", ascending=False) # 降序排序
    rank = []
    for i in range(len(rank_test_fine_tune)):
        rank.append(i + 1)
    rank_test_fine_tune["rank"] = rank
    rank_test_fine_tune.to_csv("result_format.csv",index=False)

