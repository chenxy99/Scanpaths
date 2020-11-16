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
from torchvision import transforms


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class AiR(Dataset):
    """
    get AiR data
    """

    def __init__(self,
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 bert_pretrainedQ_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 transform=None,
                 max_question_length=14,
                 label_space_length=18):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.bert_pretrainedQ_dir = bert_pretrainedQ_dir
        self.action_map = action_map
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.label_space_length = label_space_length
        self.AiR_fixations_file = join(self.AiR_fixations_dir,
                                        "AiR_fixations_" + self.type + ".json")
        self.max_question_length = max_question_length

        with open(self.AiR_fixations_file) as json_file:
            self.fixations = json.load(json_file)


    def __len__(self):
        return len(self.fixations)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def show_image_and_bboxes(self, img, objects):
        import matplotlib
        plt.figure()
        plt.imshow(img)
        ax = plt.gca()
        for index in range(len(objects)):
            object = objects[index]
            x, y, h, w = object["x"], object["y"], object["h"], object["w"]
            rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def show_image_and_bboxes_scanpath(self, img, objects, X, Y):
        import matplotlib
        plt.figure()
        plt.imshow(img)
        ax = plt.gca()
        for index in range(len(objects)):
            object = objects[index]
            x, y, h, w = object["x"], object["y"], object["h"], object["w"]
            rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.plot(X, Y, 'ro-', linewidth=2)
        plt.plot(X[0], Y[0], 'bs-', linewidth=2)
        plt.plot(X[-1], Y[-1], 'gv-', linewidth=2)
        plt.show()

    def show_image_and_fixation(self, img, x, y):
        plt.figure()
        plt.imshow(img)
        plt.plot(x, y, 'xb-')
        plt.show()

    def get_fixation(self, fix_path):
        fix_data = loadmat(fix_path)
        fixation_map = np.zeros((2 * self.resize[0], 2 * self.resize[1]), dtype=np.float32)
        for fix_id in range(fix_data["fix_x"].shape[-1]):
            x, y = fix_data["fix_x"][0, fix_id], fix_data["fix_y"][0, fix_id]
            x, y = int(x * (2 * self.resize[1] / self.origin_size[1])), int(y * (2 * self.resize[0] / self.origin_size[0]))
            fixation_map[y, x] = 1
        return fixation_map

    def get_scene_graph_info(self, fixation):
        max_object_num = 5
        img_name = fixation["image_id"]
        img_path = join(self.AiR_stimuli_dir, img_name)
        image = io.imread(img_path).astype(np.float32)

        question_annotations = fixation["annotations"]["question"]
        fullAnswer_annotations = fixation["annotations"]["fullAnswer"]
        objects = fixation["objects"]

        question_objects = [objects[name] for name in question_annotations.values()]
        fullAnswer_objects = [objects[name] for name in fullAnswer_annotations.values()]

        question_objects_pos = np.zeros((fixation["height"], fixation["width"], max_object_num), np.float32)
        fullAnswer_objects_pos = np.zeros((fixation["height"], fixation["width"], max_object_num), np.float32)
        question_objects_masks = np.zeros((max_object_num), np.float32)
        fullAnswer_objects_masks = np.zeros((max_object_num), np.float32)

        for index in range(len(question_objects)):
            object_value = question_objects[index]
            x, y, h, w = object_value["x"], object_value["y"], object_value["h"], object_value["w"]
            question_objects_pos[y:y+h, x:x+w, index] = 1
            question_objects_masks[index] = 1

        for index in range(len(fullAnswer_objects)):
            object_value = fullAnswer_objects[index]
            x, y, h, w = object_value["x"], object_value["y"], object_value["h"], object_value["w"]
            fullAnswer_objects_pos[y:y+h, x:x+w, index] = 1
            fullAnswer_objects_masks[index] = 1

        question_objects_pos_resized = resize(question_objects_pos, self.resize, anti_aliasing=True)
        fullAnswer_objects_pos_resized = resize(fullAnswer_objects_pos, self.resize, anti_aliasing=True)


        # self.show_image_and_bboxes(image/255, question_objects)
        # self.show_image_and_bboxes_scanpath(image/255, question_objects, np.array(fixation["X"]), np.array(fixation["Y"]))
        # self.show_image(question_objects_pos_resized[:, :, 0])
        return question_objects_pos_resized, fullAnswer_objects_pos_resized, \
               question_objects_masks, fullAnswer_objects_masks


    def __getitem__(self, idx):
        fixation = self.fixations[idx]
        img_name = fixation["image_id"]
        question_id = fixation["question_id"]
        img_path = join(self.AiR_stimuli_dir, img_name)

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        origin_size_y, origin_size_x = fixation["height"], fixation["width"]
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]

        scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
        # the first element denotes the termination action
        target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
        duration = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)

        pos_x = np.array(fixation["X"]).astype(np.float32)
        pos_y = np.array(fixation["Y"]).astype(np.float32)
        duration_raw = np.array(fixation["T_end"]).astype(np.float32) - np.array(fixation["T_start"]).astype(np.float32)

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

        performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"

        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, question_id + ".npy")).astype(np.float32)
        # attention_map = resize(attention_bbox, self.resize)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        # question_objects_pos, fullAnswer_objects_pos, \
        # question_objects_masks, fullAnswer_objects_masks = self.get_scene_graph_info(fixation)

        # load the labels for the questions, which project the problem and the image into a label space.
        projected_label = np.zeros((self.label_space_length), dtype=np.float32)
        for index in fixation["question_token_idx"]:
            projected_label[index] = 1
        projected_label /= projected_label.sum()

        # load bert embedding
        bert_Qlast_hidden_state = np.load(join(self.bert_pretrainedQ_dir, question_id + ".npy"))
        bert_Qsemantic = bert_Qlast_hidden_state.mean(axis=0)

        return {
            "image": image,
            "target_scanpath": target_scanpath,
            "duration": duration,
            "action_mask": action_mask,
            "duration_mask": duration_mask,
            "projected_label": projected_label,
            "attention_map": attention_map,
            "img_name": img_name,
            "question_id": question_id,
            "bert_Qsemantic": bert_Qsemantic,
            "performance": performance
        }

    def collate_func(self, batch):

        img_batch = []
        scanpath_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        projected_label_batch = []
        attention_map_batch = []
        img_name_batch = []
        question_id_batch = []
        bert_Qsemantic_batch = []
        performances_batch = []

        for sample in batch:
            tmp_img, tmp_scanpath, tmp_duration,\
            tmp_action_mask, tmp_duration_mask, \
            tmp_projected_label, tmp_attention_map, \
            tmp_img_name, tmp_question_id, tmp_bert_Qsemantic, tmp_performances =\
                sample["image"], sample["target_scanpath"], sample["duration"],\
                sample["action_mask"], sample["duration_mask"],\
                sample["projected_label"], sample["attention_map"],\
                sample["img_name"], sample["question_id"], sample["bert_Qsemantic"], sample["performance"]
            img_batch.append(tmp_img)
            scanpath_batch.append(tmp_scanpath)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            projected_label_batch.append(tmp_projected_label)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            question_id_batch.append(tmp_question_id)
            bert_Qsemantic_batch.append(tmp_bert_Qsemantic)
            performances_batch.append(tmp_performances)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["scanpaths"] = np.stack(scanpath_batch)
        data["durations"] = np.stack(duration_batch)
        data["action_masks"] = np.stack(action_mask_batch)
        data["duration_masks"] = np.stack(duration_mask_batch)
        data["attention_maps"] = np.stack(attention_map_batch)
        data["projected_labels"] = np.stack(projected_label_batch)
        data["img_names"] = img_name_batch
        data["question_ids"] = question_id_batch
        data["bert_Qsemantics"] = np.stack(bert_Qsemantic_batch)
        data["performances"] = np.stack(performances_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data


class AiR_evaluation(Dataset):
    """
    get AiR data for evaluation
    """

    def __init__(self,
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 bert_pretrainedQ_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="validation",
                 transform=None,
                 max_question_length=14,
                 label_space_length=18):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.bert_pretrainedQ_dir = bert_pretrainedQ_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.transform = transform
        self.label_space_length = label_space_length
        self.AiR_fixations_file = join(self.AiR_fixations_dir,
                                        "AiR_fixations_" + self.type + ".json")
        self.max_question_length = max_question_length

        with open(self.AiR_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.qid_to_sub = {}
        self.qid_to_img = {}
        for index, fixation in enumerate(self.fixations):
            self.qid_to_sub.setdefault(fixation['question_id'], []).append(index)
            self.qid_to_img[fixation['question_id']] = fixation['image_id']
        self.qid = list(self.qid_to_sub.keys())

    def __len__(self):
        return len(self.qid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        question_id = self.qid[idx]
        img_name = self.qid_to_img[question_id]

        img_path = join(self.AiR_stimuli_dir, img_name)

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # self.show_image(image/255)

        fix_vectors = []
        performances = []
        for ids in self.qid_to_sub[question_id]:
            fixation = self.fixations[ids]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            resizescale_x = origin_size_x / self.resize[1]
            resizescale_y = origin_size_y / self.resize[0]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = (np.array(fixation["T_end"]).astype(np.float32)
                        - np.array(fixation["T_start"]).astype(np.float32)) / 1000.0

            length = fixation["length"]

            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"
            performances.append(performance)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)


        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, question_id + ".npy")).astype(np.float32)
        # attention_map = resize(attention_bbox, self.resize)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        # load the labels for the questions, which project the problem and the image into a label space.
        projected_label = np.zeros((self.label_space_length), dtype=np.float32)
        for index in fixation["question_token_idx"]:
            projected_label[index] = 1
        projected_label /= projected_label.sum()

        # load bert embedding
        bert_Qlast_hidden_state = np.load(join(self.bert_pretrainedQ_dir, question_id + ".npy"))
        bert_Qsemantic = bert_Qlast_hidden_state.mean(axis=0)


        return {
            "image": image,
            "fix_vectors": fix_vectors,
            "performances": performances,
            "projected_label": projected_label,
            "attention_map": attention_map,
            "img_name": img_name,
            "question_id": question_id,
            "bert_Qsemantic": bert_Qsemantic,
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        projected_label_batch = []
        performances_batch = []
        attention_map_batch = []
        img_name_batch = []
        question_id_batch = []
        bert_Qsemantic_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_projected_label, tmp_attention_map, \
            tmp_img_name, tmp_performances, \
            tmp_question_id, tmp_bert_Qsemantic \
                = sample["image"], sample["fix_vectors"], sample["projected_label"], sample["attention_map"], \
                  sample["img_name"], sample["performances"], \
                  sample["question_id"], sample["bert_Qsemantic"]

            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            projected_label_batch.append(tmp_projected_label)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            performances_batch.append(tmp_performances)
            question_id_batch.append(tmp_question_id)
            bert_Qsemantic_batch.append(tmp_bert_Qsemantic)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["projected_labels"] = np.stack(projected_label_batch)
        data["attention_maps"] = np.stack(attention_map_batch)
        data["img_names"] = img_name_batch
        data["performances"] = performances_batch
        data["question_ids"] = question_id_batch
        data["bert_Qsemantics"] = np.stack(bert_Qsemantic_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data


class AiR_rl(Dataset):
    """
    get AiR data for reinforcement learning
    """

    def __init__(self,
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 bert_pretrainedQ_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="validation",
                 transform=None,
                 max_question_length=14,
                 label_space_length=18):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.bert_pretrainedQ_dir = bert_pretrainedQ_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.transform = transform
        self.label_space_length = label_space_length
        self.AiR_fixations_file = join(self.AiR_fixations_dir,
                                        "AiR_fixations_" + self.type + ".json")
        self.max_question_length = max_question_length

        with open(self.AiR_fixations_file) as json_file:
            self.fixations = json.load(json_file)

        self.qid_to_sub = {}
        self.qid_to_img = {}
        for index, fixation in enumerate(self.fixations):
            self.qid_to_sub.setdefault(fixation['question_id'], []).append(index)
            self.qid_to_img[fixation['question_id']] = fixation['image_id']
        self.qid = list(self.qid_to_sub.keys())

    def __len__(self):
        return len(self.qid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        question_id = self.qid[idx]
        img_name = self.qid_to_img[question_id]

        img_path = join(self.AiR_stimuli_dir, img_name)

        # image = io.imread(img_path).astype(np.float32)
        # image_resized = resize(image, self.resize, anti_aliasing=True)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # self.show_image(image/255)

        fix_vectors = []
        performances = []
        for ids in self.qid_to_sub[question_id]:
            fixation = self.fixations[ids]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            resizescale_x = origin_size_x / self.resize[1]
            resizescale_y = origin_size_y / self.resize[0]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = (np.array(fixation["T_end"]).astype(np.float32)
                        - np.array(fixation["T_start"]).astype(np.float32)) / 1000.0

            length = fixation["length"]

            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"
            performances.append(performance)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, question_id + ".npy")).astype(np.float32)
        # attention_map = resize(attention_bbox, self.resize)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        # load the labels for the questions, which project the problem and the image into a label space.
        projected_label = np.zeros((self.label_space_length), dtype=np.float32)
        for index in fixation["question_token_idx"]:
            projected_label[index] = 1
        projected_label /= projected_label.sum()

        # load bert embedding
        bert_Qlast_hidden_state = np.load(join(self.bert_pretrainedQ_dir, question_id + ".npy"))
        bert_Qsemantic = bert_Qlast_hidden_state.mean(axis=0)

        return {
            "image": image,
            "fix_vectors": fix_vectors,
            "performances": performances,
            "projected_label": projected_label,
            "attention_map": attention_map,
            "img_name": img_name,
            "question_id": question_id,
            "bert_Qsemantic": bert_Qsemantic,
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        projected_label_batch = []
        performances_batch = []
        attention_map_batch = []
        img_name_batch = []
        question_id_batch = []
        bert_Qsemantic_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_projected_label, tmp_attention_map, \
            tmp_img_name, tmp_performances, \
            tmp_question_id, tmp_bert_Qsemantic \
                = sample["image"], sample["fix_vectors"], sample["projected_label"], sample["attention_map"], \
                  sample["img_name"], sample["performances"], \
                  sample["question_id"], sample["bert_Qsemantic"]

            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            projected_label_batch.append(tmp_projected_label)
            attention_map_batch.append(tmp_attention_map)
            img_name_batch.append(tmp_img_name)
            performances_batch.append(tmp_performances)
            question_id_batch.append(tmp_question_id)
            bert_Qsemantic_batch.append(tmp_bert_Qsemantic)

        data = dict()
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["projected_labels"] = np.stack(projected_label_batch)
        data["attention_maps"] = np.stack(attention_map_batch)
        data["img_names"] = img_name_batch
        data["performances"] = performances_batch
        data["question_ids"] = question_id_batch
        data["bert_Qsemantics"] = np.stack(bert_Qsemantic_batch)

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
    AiR_stimuli_dir = join(data_root, "stimuli")
    AiR_fixations_dir = join(data_root, "fixations")
    AiR_dataset = AiR(AiR_stimuli_dir, AiR_fixations_dir, transform=transform)
    # test_data = AiR_dataset[0]
    for ii in range(40, 50):
        test_data = AiR_dataset[ii*10]

    train_loader = DataLoader(
        dataset=AiR_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=AiR_dataset.collate_func
    )

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
