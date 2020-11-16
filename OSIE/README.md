# Predicting Human Scanpaths in OSIE Dataset

This code implements the prediction of human scanpaths in free-viewing task.

Training your own network on OSIE dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. Hence you can directly execute the following command to train the network.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python train.py
```

## Evaluate on test split

We also provide the pretrained model in XXX, or you can use you own trained network to evaluate the performance on test split.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python test.py --evaluation_dir "./assets/pretrained_model"
```

