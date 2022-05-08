#-*- coding=utf-8 -*-
import xlrd
import pymongo

from config import check_type


def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))

def excel_table_byname(file,colnameindex=0,by_name=u'Sheet1'):#修改自己路径
     data = open_excel(file)
     table = data.sheet_by_name(by_name) #获得表格
     nrows = table.nrows  # 拿到总共行数
     colnames = table.row_values(colnameindex)  # 某一行数据 ['姓名', '用户名', '联系方式', '密码']
     list = []
     for rownum in range(1, nrows): #也就是从Excel第二行开始，第一行表头不算
         row = table.row_values(rownum)
         if row:
             app = {}
             for i in range(len(colnames)):
                 app[colnames[i]] = row[i] #表头与数据对应
             list.append(app)
     return list

# generate tasks based on model information
if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://modelmarket:Qw123456789@naibo.wang:8088/', connect=False)
    mydb = myclient['exps']
    tasks = mydb["tasks"]
    results = mydb["results"]
    task_preset = {
        "model_name": "regnety400mf",
        "original_dataset":"imagenet1k",
        "dataset": "cifar100",
        "required_image_size": 224,
        "remark": "whole",
        "split": [
            "train[:40000]",
            "train[40000:]",
            "test"
        ],
        "batch_size": 16,
        "epochs": 30,
        "GPUID": 0,
        "GPUTYPE":0, # 0 >=P100, 1>=3090, 2>=V100, 3>=A100
        "pretrained_model_path": "/home/naibo/xacc_share/models/tf-dev/feature-extractor/regnety400mf_feature_extractor_1",
        "storage_path": "/home/naibo/xacc_share/trained_models/",
        "normalize": False, # (0 keep original;1 to [0,1])
        "display_image": False,
        "random_seed": 2022,
        "learning_rate": 0.01,
        "status": "init",  # init未跑，running正在运行，done实验跑完了
    }
    tables = excel_table_byname("model_info.xls")
    for row in tables:
        task = task_preset.copy()
        for key in row:
            type = check_type(task[key])
            # print(key, row[key],task[key], type)
            if type == "int":
                task[key] = int(row[key])
            elif type == "float":
                task[key] = float(row[key])
            elif type == "bool":
                task[key] = bool(row[key])
            else:
                task[key] = row[key]
            type = check_type(task[key])
            # print(key, row[key], task[key], type)
        task_query = task.copy()
        del task_query["status"] # status状态不算重复项
        exist = len(list(tasks.find(task_query))) == 1 # 有重复项则停止插入
        if not exist: # 模型任务不存在，则
            print(task)
            tasks.insert_one(task)