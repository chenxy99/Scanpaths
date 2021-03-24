# Predicting Human Scanpaths in Visual Question Answering

This code implements the prediction of human scanpaths in three different tasks:

- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes.
- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image,
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior.

Reference
------------------
If you find the code useful in your research, please consider citing the paper.
```text
@InProceedings{xianyu:2021:scanpath,
    author={Xianyu Chen and Ming Jiang and Qi Zhao},
    title = {Predicting Human Scanpaths in Visual Question Answering},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package.  We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

Requirements
------------------

- Python 3.7
- PyTorch 1.6 (along with torchvision)

- We also provide the conda environment ``sp_baseline.yml``, you can directly run

```bash
$ conda env create -f sp_baseline.yml
```

to create the same environment where we successfully run our codes.

Tasks
------------------

We provide the corresponding codes for the aforementioned three different tasks on three different datasets.

- Visual Question Answering (AiR dataset)
- Free-viewing (OSIE dataset)

- Visual search (COCO-Search18 dataset)

We would provide more details for these tasks in their corresponding folders.