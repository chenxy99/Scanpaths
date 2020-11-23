# Predicting Human Scanpaths in COCO Search18 Dataset

This code implements the prediction of human scanpaths in visual search task.

Datasets
------------------
You can download the original data followed the [`COCO-Search18`](https://sites.google.com/view/cocosearch/home) guidance. The trainval split and the bounding box annotations for our implementation and the original implementation of inverse reinforcement learning can be download from [`link`](https://drive.google.com/drive/folders/1spD2_Eya5S5zOBO3NKILlAjMEC3_gKWc) (more detail can refer to the original [`official released code`](https://github.com/cvlab-stonybrook/Scanpath_Prediction)). For the trainval split files, you need to add `_split3` at the end of each file. We pre-process the object detector result by [`CenterNet`](https://github.com/xingyizhou/CenterNet), and alternatively, you can download it from [`link`](https://drive.google.com/file/d/1f_Ha5ppPKCngARg7_W5AlqvP6Q_N8LRu/view?usp=sharing).

The typical `<dataset_root>` should be structured as follows
```
<dataset_root>
    -- ./detectors
        -- coco_search18_detector.json              # bounding box annotation from an object detector
    -- ./fixations                                  # fixation and the training and validation splits
        coco_search18_fixations_TP_train_split3.json
        coco_search18_fixations_TP_validation_split3.json
    -- ./images                                     # image stimuli
        -- ./bottle
        -- ./bowl
        -- ......
        -- ./tv
    -- bbox_annos.npy                               # bounding box annotation for each image (available at COCO)
```

Training your own network on COCO Search18 dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./assets/`). You can also set the other hyper-parameters in `opt.py`.

- `--img_dir` Directory to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Directory to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--detector_dir` Directory to the detector results, e.g., `<dataset_root>/fixations`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning when reaching this given epoch.
- `--lambda_1` The hyper-parameter to balance the loss terms in the supervised learning stage.
- `--ablate_attention_info` To choose whether to use task guidance or not. The default parameter is `False`. If you like to ablate the task guidance, you can set it as `True`.
- `--detector_threshold` We would only use the detection results whose confidence is larger than the given `detector_threshold`.
- `--supervised_save` The default parameter is `True`. It can save a whole checkpoint before we start to use reinforcement learning to train our model. The saved checkpoint can be treated as an ablation study of self-critical sequential training. We would add `_supervised_save` as a suffix for the checkpoint document.

In the default setting, you can directly run the following command, which includes our proposed task guidance and self-critical sequential training.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py
```

If you would like to ablate the task guidance, please run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --ablate_attention_info True
```

If you would like to ablate the self-critical sequential training, you can directly find the corresponding checkpoint folders with a suffix `_supervised_save`.

## Evaluate on validation split

Since the author of COCO Search18 only releases the training and validation data, we can only evaluate the performance on the validation split.

We provide the [`pretrained model`](https://drive.google.com/file/d/1NtRD08WRTTLIpfPziImUBRqJzFMX4cH6/view?usp=sharing), and you can directly run the following command to evaluate the performance of the pretrained model on validation split.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir "./assets/pretrained_model"
```

You can also use the commands mentioned in **Training your own network on COCO Search18 dataset** to train your own network. Then you can run one of the following commands to evaluate the performance of your trained model on validation split.

If you use our default setting, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint>
```

If you ablate the task guidance in the training stage, please remember to ablate it in the evaluation stage.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint> --ablate_attention_info True
```

If you would like to evaluate the ablation of self-critical sequential training, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint + '_supervised_save'>
```

