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
import scipy.ndimage as filters
from tqdm import tqdm
from scipy.io import loadmat

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class OSIE(Dataset):
    """
    get OSIE data
    """

    def __init__(self,
                 OSIE_stimuli_dir,
                 OSIE_fixations_dir,
                 OSIE_saliencymap_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 transform=None):
        self.OSIE_stimuli_dir = OSIE_stimuli_dir
        self.OSIE_fixations_dir = OSIE_fixations_dir
        self.OSIE_saliencymap_dir = OSIE_saliencymap_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.OSIE_fixations_file = join(self.OSIE_fixations_dir,
                                        "osie_fixations_" + self.type + ".json")

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        with open(self.OSIE_fixations_file) as json_file:
            self.fixations = json.load(json_file)

    def __len__(self):
        return len(self.fixations)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def get_fixation(self, fix_path):
        fix_data = loadmat(fix_path)
        fixation_map = np.zeros((2 * self.resize[0], 2 * self.resize[1]), dtype=np.float32)
        for fix_id in range(fix_data["fix_x"].shape[-1]):
            x, y = fix_data["fix_x"][0, fix_id], fix_data["fix_y"][0, fix_id]
            x, y = int(x * (2 * self.resize[1] / self.origin_size[1])), int(y * (2 * self.resize[0] / self.origin_size[0]))
            fixation_map[y, x] = 1
        return fixation_map

    def __getitem__(self, idx):
        fixation = self.fixations[idx]
        img_name = fixation["name"]
        img_path = join(self.OSIE_stimuli_dir, img_name)
        sal_path = join(self.OSIE_saliencymap_dir, img_name)
        fix_path = join(self.OSIE_saliencymap_dir, img_name.split('.')[0] + '.mat')

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        saliency_map = io.imread(sal_path).astype(np.float32)
        saliency_map = resize(saliency_map, (2 * self.resize[0], 2 * self.resize[1]), anti_aliasing=True)
        saliency_map /= saliency_map.sum()

        fixation_map = self.get_fixation(fix_path)

        scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
        # the first element denotes the termination action
        target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
        duration = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)

        pos_x = np.array(fixation["X"]).astype(np.float32)
        pos_y = np.array(fixation["Y"]).astype(np.float32)
        duration_raw = np.array(fixation["T"]).astype(np.float32)

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

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": image,
            "saliency_map": saliency_map,
            "fixation_map": fixation_map,
            "target_scanpath": target_scanpath,
            "duration": duration,
            "action_mask": action_mask,
            "duration_mask": duration_mask,
            "img_name": img_name,
        }

    def collate_func(self, batch):

        img_batch = []
        saliency_map_batch = []
        fixation_map_batch = []
        scanpath_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        img_name_batch = []

        for sample in batch:
            tmp_img, tmp_saliency_map, tmp_fixation_map, tmp_scanpath, tmp_duration,\
            tmp_action_mask, tmp_duration_mask, tmp_img_name =\
                sample["image"], sample["saliency_map"], sample["fixation_map"],\
                sample["target_scanpath"], sample["duration"],\
                sample["action_mask"], sample["duration_mask"], sample["img_name"]
            img_batch.append(tmp_img)
            saliency_map_batch.append(tmp_saliency_map)
            fixation_map_batch.append(tmp_fixation_map)
            scanpath_batch.append(tmp_scanpath)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            img_name_batch.append(tmp_img_name)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["saliency_maps"] = np.stack(saliency_map_batch)
        data["fixation_maps"] = np.stack(fixation_map_batch)
        data["scanpaths"] = np.stack(scanpath_batch)
        data["durations"] = np.stack(duration_batch)
        data["action_masks"] = np.stack(action_mask_batch)
        data["duration_masks"] = np.stack(duration_mask_batch)
        data["img_names"] = img_name_batch

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data


class OSIE_evaluation(Dataset):
    """
    get OSIE data for evaluation
    """

    def __init__(self,
                 OSIE_stimuli_dir,
                 OSIE_fixations_dir,
                 OSIE_saliencymap_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.OSIE_stimuli_dir = OSIE_stimuli_dir
        self.OSIE_fixations_dir = OSIE_fixations_dir
        self.OSIE_saliencymap_dir = OSIE_saliencymap_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform
        self.OSIE_fixations_file = join(self.OSIE_fixations_dir,
                                        "osie_fixations_" + self.type + ".json")


        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        with open(self.OSIE_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(fixation['name'], []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.OSIE_stimuli_dir, img_name)
        sal_path = join(self.OSIE_saliencymap_dir, img_name)

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        saliency_map = io.imread(sal_path).astype(np.float32)
        saliency_map = resize(saliency_map, (2 * self.resize[0], 2 * self.resize[1]), anti_aliasing=True)
        saliency_map /= saliency_map.sum()

        fix_vectors = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

        return {
            "image": image,
            "saliency_map": saliency_map,
            "fix_vectors": fix_vectors,
            "img_name": img_name
        }

    def collate_func(self, batch):

        img_batch = []
        saliency_map_batch = []
        fix_vectors_batch = []
        img_name_batch = []

        for sample in batch:
            tmp_img, tmp_saliency_map, tmp_fix_vectors, tmp_img_name \
                = sample["image"], sample["saliency_map"], sample["fix_vectors"], sample["img_name"]
            img_batch.append(tmp_img)
            saliency_map_batch.append(tmp_saliency_map)
            fix_vectors_batch.append(tmp_fix_vectors)
            img_name_batch.append(tmp_img_name)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["saliency_maps"] = np.stack(saliency_map_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["img_names"] = img_name_batch

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data


class OSIE_rl(Dataset):
    """
    get OSIE data for reinforcement learning
    """

    def __init__(self,
                 OSIE_stimuli_dir,
                 OSIE_fixations_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.OSIE_stimuli_dir = OSIE_stimuli_dir
        self.OSIE_fixations_dir = OSIE_fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform
        self.OSIE_fixations_file = join(self.OSIE_fixations_dir,
                                        "osie_fixations_" + self.type + ".json")

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        with open(self.OSIE_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(fixation['name'], []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        # idx //= 15
        img_name = self.imgid[idx]
        img_path = join(self.OSIE_stimuli_dir, img_name)

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        fix_vectors = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

        return {
            "image": image,
            "fix_vectors": fix_vectors,
            "img_name": img_name,
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        img_name_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_img_name = sample["image"], sample["fix_vectors"], sample["img_name"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            img_name_batch.append(tmp_img_name)

        data = {}
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["img_names"] = img_name_batch

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data

if __name__ == "__main__":
    data_root = "../data"
    OSIE_stimuli_dir = join(data_root, "stimuli")
    OSIE_fixations_dir = join(data_root, "fixations")
    osie_dataset = OSIE(OSIE_stimuli_dir, OSIE_fixations_dir)
    test_data = osie_dataset[0]

    train_loader = DataLoader(
        dataset=osie_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=osie_dataset.collate_func
    )

    osie_val = OSIE_evaluation(OSIE_stimuli_dir, OSIE_fixations_dir, type="validation")

    val_loader = DataLoader(
        dataset=osie_val,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=osie_val.collate_func
    )


    for i_batch, batch in tqdm(enumerate(osie_val)):
        pass
