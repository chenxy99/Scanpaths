# Predicting Human Scanpaths in AiR Dataset

This code implements the prediction of human scanpaths in visual question answering task.

Datasets
------------------

This dataset is mainly based on [`AiR: Attention with Reasoning Capability`](https://www-users.cs.umn.edu/~qzhao/air.html) as well as [`GQA`](https://cs.stanford.edu/people/dorarad/gqa/download.html). You can download the above dataset from the provided links. Then you can get the splits of this dataset by execution of the following command 

```bash
$ python ./preprocess/preprocess_fixations.py
```

The machine attention from AiR dataset can be preprocessed by the released code [`Air`](https://github.com/szzexpoi/AiR). Alternatively, we provide the pre-processed [`fixation files`](https://drive.google.com/file/d/17q7lTvAMejyR48BNlE6vVYSPCwvo_6sI/view?usp=sharing), ['stimuli files'](https://drive.google.com/file/d/1Dyi0y6ktSSwthhU90uOmM1fkrptAdzJK/view?usp=sharing) as well as the [`machine attention files`](https://drive.google.com/file/d/1mpeLq_nORcOW4GKXwpjWzgJaMkHJC9KX/view?usp=sharing), and therefore you can directly download them.

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

We have set all the corresponding hyper-parameters in ``opt.py``. Hence you can directly execute the following command to train the network.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python train.py
```

## Evaluate on test split

We also provide the [`pretrained model`](https://drive.google.com/file/d/1rvQwMW83g1lZOpWYy-8Iis_qrYQr3sbO/view?usp=sharing), or you can use your own trained network to evaluate the performance on test split.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python test.py --evaluation_dir "./assets/pretrained_model"
```

