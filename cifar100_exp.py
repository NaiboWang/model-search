import tensorflow as tf
import tensorflow_datasets as tfds

from finetuner import Finetuner

if __name__ == '__main__':
    # import os
    # print(os.environ)
    # Save class_names and n_classes for later
    dataset = "cifar100"
    split = ["train[:40000]", "train[40000:]", "test"]
    # ds, ds_info = tfds.load(name=dataset,
    #                         split=split,
    #                         shuffle_files=True,
    #                         as_supervised=True,
    #                         with_info=True)
    # train_set, valid_set, test_set = ds[0], ds[1], ds[2]
    # # class_names = info.features["label"].names
    # # n_classes = info.features["label"].num_classes
    # print("Train set size: ", len(train_set))
    # print("Valid set size: ", len(valid_set))
    # print("Test set size: ", len(test_set))
    # print(ds_info.features["label"].num_classes)
    finetuner = Finetuner()
    finetuner.finetune(dataset="cifar100",epochs=20,batch_size=16,split=split)
