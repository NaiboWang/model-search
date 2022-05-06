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
    strategy = tf.distribute.MirroredStrategy(
        devices=["/physical_device:GPU:1", "/physical_device:GPU:2","/physical_device:GPU:3","/physical_device:GPU:4","/physical_device:GPU:5","/physical_device:GPU:6","/physical_device:GPU:7", ])
    with strategy.scope():
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        finetuner = Finetuner()
        finetuner.finetune(dataset="cifar100",epochs=20,batch_size=16,split=split)
