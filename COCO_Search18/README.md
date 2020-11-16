# Predicting Human Scanpaths in COCO Search18 Dataset

This code implements the prediction of human scanpaths in visual search task.

Training your own network on COCO Search18 dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. Hence you can directly execute the following command to train the network.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python train.py
```

## Evaluate on validation split

Since the author of COCO Search18 only releases the training and validation data, we can only evaluate the performance on the validation split.

We provide the pretrained model in XXX, or you can use you own trained network to evaluate the performance on validation split.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python test.py --evaluation_dir "./assets/pretrained_model"
```

