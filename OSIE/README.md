# Predicting Human Scanpaths in OSIE Dataset

This code implements the prediction of human scanpaths in free-viewing task.

Datasets
------------------

This dataset is mainly based on [`predicting-human-gaze-beyond-pixels`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels). You need to download [`stimuli`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/stimuli) and [`fixations`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/eye) and put them in a proper location. Then you can get the splits of this dataset by the execution of the following command 

```bash
$ python ./preprocess/preprocess_fixations.py
```

Alternatively, we provide the pre-processed fixation files, and you can directly download them from [`link`](https://drive.google.com/file/d/1p2hf85w22RvZjk1n2VeVY0EgT50rfQJC/view?usp=sharing).

The typical `<dataset_root>` should be structured as follows

```
<dataset_root>
    -- ./fixations                                  # fixation and the training, validation and test splits
        osie_fixations_test.json
        osie_fixations_train.json
        osie_fixations_validation.json
    -- ./stimuli                                     # image stimuli
```

Training your own network on OSIE dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./assets/`). You can also set the other hyper-parameters in `opt.py`.

- `--img_dir` Directory to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Directory to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning when reaching this given epoch.
- `--lambda_1` The hyper-parameter to balance the loss terms in the supervised learning stage.
- `--supervised_save` The default parameter is `True`. It can save a whole checkpoint before we start to use reinforcement learning to train our model. The saved checkpoint can be treated as an ablation study of self-critical sequential training. We would add `_supervised_save` as a suffix for the checkpoint document.

In the default setting, you can directly run the following command, which includes our proposed self-critical sequential training.

If you would like to ablate the self-critical sequential training, you can directly find the corresponding checkpoint folders with a suffix `_supervised_save`.

## Evaluate on test split

We also provide the [`pretrained model`](https://drive.google.com/file/d/1SWH3w3XTX_i7bkY3YMTAXV2IKwEnDH5S/view?usp=sharing), and you can directly run the following command to evaluate the performance of the pretrained model on test split.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir "./assets/pretrained_model"
```

You can also use the commands mentioned in **Training your own network on OSIE dataset** to train your own network. Then you can run one of the following commands to evaluate the performance of your trained model on test split.

If you use our default setting, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint>
```

If you would like to evaluate the ablation of self-critical sequential training, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint + '_supervised_save'>
```

