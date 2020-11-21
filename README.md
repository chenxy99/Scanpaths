# Predicting Human Scanpaths in Visual Question Answering

This code implements the prediction of human scanpaths in three different tasks (visual question answering task, free-viewing task and visual search task).

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for CVPR 2021, paper ID 443.
```

Disclaimer
------------------
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate in our pipeline.

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

We provide the corresponding codes for the mentioned three different tasks.

- Visual Question Answering (AiR)
- Free-viewing (OSIE)

- Visual search (COCO_Search18)

More details can refer to the corresponding documents.