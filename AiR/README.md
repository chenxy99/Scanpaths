# Predicting Human Scanpaths in AiR Dataset

This code implements the prediction of human scanpaths in visual question answering task.

Datasets
------------------

This dataset is mainly based on [`AiR: Attention with Reasoning Capability`](https://www-users.cs.umn.edu/~qzhao/air.html) as well as [`GQA`](https://cs.stanford.edu/people/dorarad/gqa/download.html). You can download the above dataset from the provided links. Then you can get the splits of this dataset by execution of the following command 

```bash
$ python ./preprocess/preprocess_fixations.py
```

The machine attention from AiR dataset can be preprocessed by the released code [`Air`](https://github.com/szzexpoi/AiR). Alternatively, we provide the pre-processed [`fixation files`](https://drive.google.com/file/d/17q7lTvAMejyR48BNlE6vVYSPCwvo_6sI/view?usp=sharing), [`stimuli files`](https://drive.google.com/file/d/1Dyi0y6ktSSwthhU90uOmM1fkrptAdzJK/view?usp=sharing) as well as the [`machine attention files`](https://drive.google.com/file/d/1mpeLq_nORcOW4GKXwpjWzgJaMkHJC9KX/view?usp=sharing), and therefore you can directly download them.

The typical `<dataset_root>` should be structured as follows

```
<dataset_root>
    -- ./attention_reasoning                        # machine attention from AiR
    -- ./fixations                                  # fixation and the training, validation and test splits
        AiR_fixations_test.json
        AiR_fixations_train.json
        AiR_fixations_validation.json
    -- ./stimuli                                    # image stimuli
```

Training your own network on AiR dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./assets/`). You can set the other hyper-parameters in `opt.py`.

- `--img_dir` Directory to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Directory to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--att_dir` Directory to the attention maps, e.g., `<dataset_root>/attention_reasoning`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning when reach this given epoch.
- `--lambda_1` The hyper-parameter to balance the loss terms in supervised learning stage.
- `--lambda_5` The hyper-parameter to control the contribution of the Consistency-Divergence loss. For the propose of ablation study of Consistency-Divergence loss, you can directly set it `0` to ablate the  Consistency-Divergence loss.
- `--ablate_attention_info` To choose whether to use the task guidance or not. The default parameter is `False`. If you like to ablate the task guidance, you can set it as `True`.
- `--supervised_save` The default parameter is `True`. It can save a whole checkpoint before we start to use reinforcement learning. The saved checkpoint can be treated as an ablation study of self-critical sequential training. We would add `_supervised_save` as a suffix for the checkpoint document.

In the default setting, you can directly run the following command which including our proposed task guidance, self-critical sequential training and Consistency-Divergence loss.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py
```

If you would like to ablate the task guidance, please run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --ablate_attention_info True
```

If you would like to ablate the Consistency-Divergence loss, please run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --lambda_5 0
```

If you would like to ablate the self-critical sequential training, you can directly find the corresponding checkpoint document with a suffix `_supervised_save`.

## Evaluate on test split

We also provide the [`pretrained model`](https://drive.google.com/file/d/1rvQwMW83g1lZOpWYy-8Iis_qrYQr3sbO/view?usp=sharing), and you can directly run the following command to evaluate the performance of pretrained model on test split.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir "./assets/pretrained_model"
```

You can also use the commands mentioned in **Training your own network on AiR dataset** to train your own network. Then you can run one of the following commands to evaluate the performance of your trained model on test split.

If you use our default setting, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint>
```

If you ablate the task guidance in the training stage, please remember to ablate it in evaluation stage.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint> --ablate_attention_info True
```

If you have ablated the Consistency-Divergence loss, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint>
```

If you would like to evaluate the ablation of self-critical sequential training, you can run the following command.

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir <your_checkpoint + `_supervised_save`>
```

