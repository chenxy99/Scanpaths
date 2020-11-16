import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from os.path import join
import json
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as filters
from tqdm import tqdm
from scipy.io import loadmat
from torchvision import transforms


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

epsilon = 1e-7


class COCO_Search18(Dataset):
    """
    get COCO Search18 data
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 COCO_Search18_saliencymap_dir=None,
                 action_map=(30, 40),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):
        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.COCO_Search18_saliencymap_dir = COCO_Search18_saliencymap_dir
        self.action_map = action_map
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.split = split
        self.transform = transform
        self.saliency_map_blur_sigma = saliency_map_blur_sigma
        self.detector_threshold = detector_threshold
        self.COCO_Search18_fixations_file = join(self.COCO_Search18_fixations_dir,
                                        "coco_search18_fixations_TP_" + self.type + "_" + self.split + ".json")
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")
        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        with open(self.COCO_Search18_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index]["score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

        # duration_length = np.zeros((len(self.fixations), 1), np.float32)
        # for index in range(len(self.fixations)):
        #     fixation = self.fixations[index]
        #     duration_length[index] = fixation["length"]
        # sns.distplot(np.array(duration_length))
        # plt.show()


    def __len__(self):
        return len(self.fixations)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def show_image_and_fixation(self, img, x, y):
        plt.figure()
        plt.imshow(img)
        plt.plot(x, y, 'xb-')
        plt.show()

    # def get_fixation(self, fix_path):
    #     fix_data = loadmat(fix_path)
    #     fixation_map = np.zeros((2 * self.resize[0], 2 * self.resize[1]), dtype=np.float32)
    #     for fix_id in range(fix_data["fix_x"].shape[-1]):
    #         x, y = fix_data["fix_x"][0, fix_id], fix_data["fix_y"][0, fix_id]
    #         x, y = int(x * (2 * self.resize[1] / self.origin_size[1])), int(y * (2 * self.resize[0] / self.origin_size[0]))
    #         fixation_map[y, x] = 1
    #     return fixation_map

    def get_fixation(self, pos_x, pos_y, duration_raw, fixation):
        fixation_map = np.zeros((2 * self.resize[0], 2 * self.resize[1]), dtype=np.float32)
        origin_size_y, origin_size_x = fixation["height"], fixation["width"]
        for fix_id in range(pos_x.shape[0]):
            x, y = pos_x[fix_id], pos_y[fix_id]
            x, y = int(x * (2 * self.resize[1] / origin_size_x)), int(y * (2 * self.resize[0] / origin_size_y))
            fixation_map[y, x] = duration_raw[fix_id] / 1000
        saliency_map = filters.gaussian_filter(fixation_map, 30)
        saliency_map /= (saliency_map.sum() + epsilon)
        return fixation_map, saliency_map

    def extract_scanpath_info(self, fixation_sample):
        scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
        # the first element denotes the termination action
        target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
        duration = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)

        pos_x = np.array(fixation_sample["X"]).astype(np.float32)
        pos_x[pos_x >= self.action_map[1] * self.downscale_x] = self.action_map[1] * self.downscale_x - 1
        pos_y = np.array(fixation_sample["Y"]).astype(np.float32)
        pos_y[pos_y >= self.action_map[0] * self.downscale_y] = self.action_map[0] * self.downscale_y - 1
        duration_raw = np.array(fixation_sample["T"]).astype(np.float32)

        pos_x_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
        pos_y_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
        for index in range(len(pos_x)):
            # only preserve the max_length ground-truth
            if index == self.max_length:
                break
            pos_x_discrete[index] = (pos_x[index] / self.downscale_x).astype(np.int32)
            pos_y_discrete[index] = (pos_y[index] / self.downscale_y).astype(np.int32)
            duration[index] = duration_raw[index] / 1000.0
            action_mask[index] = 1
            duration_mask[index] = 1
        if action_mask.sum() <= self.max_length - 1:
            action_mask[int(action_mask.sum())] = 1

        for index in range(self.max_length):
            if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
                target_scanpath[index, 0] = 1
            else:
                scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
                if self.blur_sigma:
                    scanpath[index] = filters.gaussian_filter(scanpath[index], self.blur_sigma)
                    scanpath[index] /= scanpath[index].sum()
                target_scanpath[index, 1:] = scanpath[index].reshape(-1)

        # fixation_map, saliency_map = self.get_fixation(pos_x, pos_y, duration_raw, fixation_sample)

        return target_scanpath, duration, action_mask, duration_mask


    def __getitem__(self, idx):
        fixation = self.fixations[idx]
        img_name = fixation["name"]
        task = fixation["task"]
        img_path = join(join(self.COCO_Search18_stimuli_dir, task), img_name)

        image_id = img_name.split(".")[0]

        origin_image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        if self.transform is not None:
            image = self.transform(image)
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]

        scanpath, duration, action_mask, duration_mask = self.extract_scanpath_info(fixation)

        task = self.name2int[task]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id, []):
            if det["category"] == fixation["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        return {
            "image": image,
            "scanpath": scanpath,
            "duration": duration,
            "action_mask": action_mask,
            "duration_mask": duration_mask,
            "attention_map": attention_map,
            "img_name": img_name,
            "task": task
        }

    def collate_func(self, batch):

        img_batch = []
        scanpath_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        attention_map_batch = []
        img_name_batch = []
        task_batch = []

        for sample in batch:
            tmp_img, tmp_scanpath, tmp_duration,\
            tmp_action_mask, tmp_duration_mask,\
            tmp_attention_map, tmp_img_name, tmp_task =\
                sample["image"], sample["scanpath"], sample["duration"],\
                sample["action_mask"], sample["duration_mask"], \
                sample["attention_map"], sample["img_name"], sample["task"]
            img_batch.append(tmp_img)
            scanpath_batch.append(tmp_scanpath)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["scanpaths"] = np.stack(scanpath_batch)
        data["durations"] = np.stack(duration_batch)
        data["action_masks"] = np.stack(action_mask_batch)
        data["duration_masks"] = np.stack(duration_mask_batch)
        data["attention_maps"] = np.stack(attention_map_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = np.stack(task_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data


class COCO_Search18_evaluation(Dataset):
    """
    get COCO_Search18 data for evaluation
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 COCO_Search18_saliencymap_dir = None,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="validation",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):

        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.COCO_Search18_saliencymap_dir = COCO_Search18_saliencymap_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.split = split
        self.transform = transform
        self.detector_threshold = detector_threshold
        self.COCO_Search18_fixations_file = join(self.COCO_Search18_fixations_dir,
                                        "coco_search18_fixations_TP_" + self.type + "_" + self.split + ".json")
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")

        with open(self.COCO_Search18_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        self.fixations_dict = dict()
        for index in range(len(self.fixations)):
            fixation = self.fixations[index]
            task = fixation["task"]
            img_name = fixation["name"]
            task_dict = self.fixations_dict.setdefault(task, dict())
            task_dict.setdefault(img_name, []).append(fixation)
        self.fixations_list = list()
        for _, task_value in self.fixations_dict.items():
            for key, value in task_value.items():
                self.fixations_list.append(value)

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index]["score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

    def __len__(self):
        return len(self.fixations_list)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        fixations = self.fixations_list[idx]
        img_name = fixations[0]["name"]
        # if img_name == "000000211326.jpg":
        #     a=1
        task = fixations[0]["task"]
        img_path = join(join(self.COCO_Search18_stimuli_dir, task), img_name)

        image_id = img_name.split(".")[0]

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        resizescale_x = origin_size_x / self.resize[1]
        resizescale_y = origin_size_y / self.resize[0]
        if self.transform is not None:
            image = self.transform(image)
        # self.show_image(image/255)

        fix_vectors = []
        for ids in range(len(fixations)):
            fixation = fixations[ids]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

        task = self.name2int[task]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id, []):
            if det["category"] == fixation["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        return {
            "image": image,
            "fix_vectors": fix_vectors,
            "attention_map": attention_map,
            "img_name": img_name,
            "task": task,
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        attention_map_batch = []
        img_name_batch = []
        task_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_attention_map, tmp_img_name, tmp_task, \
                = sample["image"], sample["fix_vectors"], sample["attention_map"], sample["img_name"], sample["task"]

            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["attention_maps"] = np.stack(attention_map_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = np.stack(task_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data


class COCO_Search18_rl(Dataset):
    """
    get COCO_Search18 data for reinforcement learning
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 COCO_Search18_saliencymap_dir = None,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="train",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):

        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.COCO_Search18_saliencymap_dir = COCO_Search18_saliencymap_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.split = split
        self.transform = transform
        self.detector_threshold = detector_threshold
        self.COCO_Search18_fixations_file = join(self.COCO_Search18_fixations_dir,
                                        "coco_search18_fixations_TP_" + self.type + "_" + self.split + ".json")
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")

        with open(self.COCO_Search18_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        self.fixations_dict = dict()
        for index in range(len(self.fixations)):
            fixation = self.fixations[index]
            task = fixation["task"]
            img_name = fixation["name"]
            task_dict = self.fixations_dict.setdefault(task, dict())
            task_dict.setdefault(img_name, []).append(fixation)
        self.fixations_list = list()
        for _, task_value in self.fixations_dict.items():
            for key, value in task_value.items():
                self.fixations_list.append(value)

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index]["score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

    def __len__(self):
        return len(self.fixations_list)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        fixations = self.fixations_list[idx]
        img_name = fixations[0]["name"]
        task = fixations[0]["task"]
        img_path = join(join(self.COCO_Search18_stimuli_dir, task), img_name)

        image_id = img_name.split(".")[0]

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        resizescale_x = origin_size_x / self.resize[1]
        resizescale_y = origin_size_y / self.resize[0]
        if self.transform is not None:
            image = self.transform(image)
        # self.show_image(image/255)

        fix_vectors = []
        for ids in range(len(fixations)):
            fixation = fixations[ids]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

        task = self.name2int[task]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id, []):
            if det["category"] == fixation["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        return {
            "image": image,
            "fix_vectors": fix_vectors,
            "attention_map": attention_map,
            "img_name": img_name,
            "task": task,
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        attention_map_batch = []
        img_name_batch = []
        task_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_attention_map, tmp_img_name, tmp_task, \
                = sample["image"], sample["fix_vectors"], sample["attention_map"], sample["img_name"], sample["task"]

            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["attention_maps"] = np.stack(attention_map_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = np.stack(task_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data

if __name__ == "__main__":
    data_root = "../data"
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    COCO_Search18_stimuli_dir = join(data_root, "images")
    COCO_Search18_fixations_dir = join(data_root, "fixations")
    COCO_Search18_dataset = COCO_Search18(COCO_Search18_stimuli_dir, COCO_Search18_fixations_dir,
                                          None, transform=transform)
    # test_data = AiR_dataset[0]
    for ii in range(40, 50):
        train_data = COCO_Search18_dataset[ii*10]

    train_loader = DataLoader(
        dataset=COCO_Search18_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=COCO_Search18_dataset.collate_func
    )

    COCO_Search18_val = COCO_Search18_evaluation(COCO_Search18_stimuli_dir, COCO_Search18_fixations_dir,
                                                 None, transform=transform)

    val_loader = DataLoader(
        dataset=COCO_Search18_val,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=COCO_Search18_val.collate_func
    )
    # test_data = AiR_dataset[0]
    for ii in range(40, 50):
        train_data = COCO_Search18_val[ii]

    for i_batch, batch in tqdm(enumerate(val_loader)):
        pass


    for i_batch, batch in tqdm(enumerate(train_loader)):
        pass

    AiR_val = AiR_evaluation(AiR_stimuli_dir, AiR_fixations_dir, type="validation", transform=transform)

    test_data = AiR_val[0]

    val_loader = DataLoader(
        dataset=AiR_val,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=AiR_val.collate_func
    )


    for i_batch, batch in tqdm(enumerate(val_loader)):
        pass
