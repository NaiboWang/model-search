from functools import reduce

import pymongo
import csv


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
    row_title2.append("testSet_before_transfer_learning")
    row_title2.append("validationSet_before_transfer_learning")
    row_title2.append("trainingSet_before_transfer_learning")
    row_title2.append("testSet_after_transfer_learning")
    row_title2.append("validationSet_after_transfer_learning")
    row_title2.append("trainingSet_after_transfer_learning")
    row_title2.append("testSet_before_fine_tune")
    row_title2.append("validationSet_before_fine_tune")
    row_title2.append("trainingSet_before_fine_tune")
    row_title2.append("testSet_after_fine_tune")
    row_title2.append("validationSet_after_fine_tune")
    row_title2.append("trainingSet_after_fine_tune")

    row_titles = []
    for key in row_title:
        row_titles.append(key)
    for key in row_title2:
        row_titles.append(key)

    rows = []
    for result in exp_results:
        row = []
        for key in row_title:
            row.append(str(result[key]))
        for key in row_title2:
            accuracy = reduce(lambda pre, cur: cur if cur['index'] == key else pre, result["results"],
                                 None)["accuracy"]  # 对象数组查找
            row.append(str(accuracy))
        print(row)
        rows.append(row)
    with open("result_format.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_titles)
        for r in rows:
            writer.writerow(r)