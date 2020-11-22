# Predicting Human Scanpaths in OSIE Dataset

This code implements the prediction of human scanpaths in free-viewing task.

Datasets
------------------

This dataset is mainly developed by [`predicting-human-gaze-beyond-pixels`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels). You need to download [`stimuli`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/stimuli) and [`fixations`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/eye) and put them in a proper location. Then you can get the splits of this dataset by execution of the following command 

```bash
$ python ./preprocess/pre[rpcess_fixations.py
```

Alternatively, we provide the pre-processed fixation files and you can directly download it from [`link`](https://drive.google.com/file/d/1p2hf85w22RvZjk1n2VeVY0EgT50rfQJC/view?usp=sharing).

The typical `<dataset_root>` should be structured as follows

```
<dataset_root>
    -- ./fixations                                  # fixation and the training, validation and test splits
        osie_fixations_test.json
        osie_fixations_train.json
        osie_fixations_validation.json
    -- ./images                                     # image stimuli
```

Training your own network on OSIE dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. Hence you can directly execute the following command to train the network.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python train.py
```

## Evaluate on test split

We also provide the [`pretrained model`](https://drive.google.com/file/d/121Liw1H2kT3vZpWlZ2q_Dlo-6SW9FxL8/view?usp=sharing), or you can use you own trained network to evaluate the performance on test split.

```bash
$ CUDA_VISIBLE_DEVICES=0, 1 python test.py --evaluation_dir "./assets/pretrained_model"
```

