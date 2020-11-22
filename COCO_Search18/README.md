# Predicting Human Scanpaths in COCO Search18 Dataset

This code implements the prediction of human scanpaths in visual search task.

Datasets
------------------
You can download the original data followed the [`COCO-Search18`](https://sites.google.com/view/cocosearch/home) guidance. The trainval split and the bounding box annotations for our implementation and original implementation of inverse reinforcement learning can be download from  [`link`](https://drive.google.com/drive/folders/1spD2_Eya5S5zOBO3NKILlAjMEC3_gKWc) (more detail can refer to the original [`official released code`](https://github.com/cvlab-stonybrook/Scanpath_Prediction)). For the trainval split files, you need to add `_split3` at the end of each file. We pre-process the object detector result by [`CenterNet`](https://github.com/xingyizhou/CenterNet) and you can download it from [`link`](https://drive.google.com/file/d/1f_Ha5ppPKCngARg7_W5AlqvP6Q_N8LRu/view?usp=sharing).

The typical `<dataset_root>` should be structured as follows
```
<dataset_root>
	-- ./detectors
		-- coco_search18_detector.json				 # bounding box annotation from an object detector
	-- ./fixations
		coco_search18_fixations_TP_train_split3.json
		coco_search18_fixations_TP_validation_split3.json
	-- ./images										 # image stimuli
		-- ./bottle
		-- ./bowl
		-- ......
		-- ./tv
    -- bbox_annos.npy                                # bounding box annotation for each image (available at COCO)
```



One can follow the instructions in [data/README.md](data/README.md) to create the corresponding data. More specifically, we download the preprocessed file or preextracted features from [link](https://drive.google.com/drive/folders/1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).
You need to download as least the following files, unzip them and put them in the `data` folder.

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

