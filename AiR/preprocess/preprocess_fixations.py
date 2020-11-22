import json
import os
import numpy as np
import scipy.io as sio
from skimage import io

consolidated_answer_json_file = "./data/air-fixations/consolidated_answers.json"
val_balanced_questions_json_file = "./data/questions1.2/val_balanced_questions.json"
val_scene_graph_json_file = "./data/sceneGraphs/val_sceneGraphs.json"
save_stimuli_file = "./data/stimuli"

with open(consolidated_answer_json_file) as json_file:
    consolidated_answer = json.load(json_file)

with open(val_balanced_questions_json_file) as json_file:
    val_balanced_questions = json.load(json_file)

with open(val_scene_graph_json_file) as json_file:
    val_scene_graph = json.load(json_file)

qid = list(consolidated_answer["accuracy"])
image_ids = {id: val_balanced_questions[id]["imageId"] for id in qid}

gqa_images_folder = "./GQA/images"
gqa_images_names = os.listdir(gqa_images_folder)

np.random.seed(0)
np.random.shuffle(qid)
length = len(qid)
train, val, test = qid[: int(length * .8)], qid[int(length * .8) : int(length * .9)], qid[int(length * .9):]

except_list = list()
all_fixation_length = list()

train_list = list()
for qid_value in train:
    fix_dir = os.path.join("./data/air-fixations/fix", qid_value)
    fix_files = os.listdir(fix_dir)
    for fix_file in fix_files:
        example_dict = dict(val_balanced_questions[qid_value])
        fix_data = sio.loadmat(os.path.join(fix_dir, fix_file))
        img_id = image_ids[qid_value] + ".jpg"
        subject = fix_file.split(".")[0]
        example_dict["image_id"] = img_id
        example_dict["subject"] = subject
        example_dict["question_id"] = qid_value

        image = io.imread(os.path.join(gqa_images_folder, img_id)).astype(np.float32)
        H, W = image.shape[0], image.shape[1]
        example_dict["height"] = H
        example_dict["width"] = W

        if fix_data["xy"].shape[0] != 0:
            example_dict["X"] = fix_data["xy"][:, 0].tolist()
            example_dict["Y"] = fix_data["xy"][:, 1].tolist()
            example_dict["T_start"] = fix_data["t"][:, 0].tolist()
            example_dict["T_end"] = fix_data["t"][:, 1].tolist()

            example_dict["length"] = fix_data["t"].shape[0]
            all_fixation_length.append(example_dict["length"])
        else:
            except_list.append(fix_dir + "    " + subject)
            print(fix_dir + "    " + subject)
            break

        example_dict["subject_answer"] = consolidated_answer[subject][qid_value]
        example_dict["accuracy"] = consolidated_answer["accuracy"][qid_value]

        example_dict["split"] = "train"

        question = consolidated_answer["question"][qid_value]
        answer = consolidated_answer["answer"][qid_value]

        scene_graph_info = val_scene_graph[image_ids[qid_value]]

        example_dict["objects"] = scene_graph_info["objects"]

        train_list.append(example_dict)


validation_list = list()
for qid_value in val:
    fix_dir = os.path.join("./data/air-fixations/fix", qid_value)
    fix_files = os.listdir(fix_dir)
    for fix_file in fix_files:
        example_dict = dict(val_balanced_questions[qid_value])
        fix_data = sio.loadmat(os.path.join(fix_dir, fix_file))
        img_id = image_ids[qid_value] + ".jpg"
        subject = fix_file.split(".")[0]
        example_dict["image_id"] = img_id
        example_dict["subject"] = subject
        example_dict["question_id"] = qid_value

        image = io.imread(os.path.join(gqa_images_folder, img_id)).astype(np.float32)
        H, W = image.shape[0], image.shape[1]
        example_dict["height"] = H
        example_dict["width"] = W

        if fix_data["xy"].shape[0] != 0:
            example_dict["X"] = fix_data["xy"][:, 0].tolist()
            example_dict["Y"] = fix_data["xy"][:, 1].tolist()
            example_dict["T_start"] = fix_data["t"][:, 0].tolist()
            example_dict["T_end"] = fix_data["t"][:, 1].tolist()

            example_dict["length"] = fix_data["t"].shape[0]
            all_fixation_length.append(example_dict["length"])
        else:
            except_list.append(fix_dir + "    " + subject)
            print(fix_dir + "    " + subject)
            break

        example_dict["subject_answer"] = consolidated_answer[subject][qid_value]
        example_dict["accuracy"] = consolidated_answer["accuracy"][qid_value]

        example_dict["split"] = "validation"

        question = consolidated_answer["question"][qid_value]
        answer = consolidated_answer["answer"][qid_value]

        scene_graph_info = val_scene_graph[image_ids[qid_value]]

        example_dict["objects"] = scene_graph_info["objects"]

        validation_list.append(example_dict)

test_list = list()
for qid_value in test:
    fix_dir = os.path.join("./data/air-fixations/fix", qid_value)
    fix_files = os.listdir(fix_dir)
    for fix_file in fix_files:
        example_dict = dict(val_balanced_questions[qid_value])
        fix_data = sio.loadmat(os.path.join(fix_dir, fix_file))
        img_id = image_ids[qid_value] + ".jpg"
        subject = fix_file.split(".")[0]
        example_dict["image_id"] = img_id
        example_dict["subject"] = subject
        example_dict["question_id"] = qid_value

        image = io.imread(os.path.join(gqa_images_folder, img_id)).astype(np.float32)
        H, W = image.shape[0], image.shape[1]
        example_dict["height"] = H
        example_dict["width"] = W

        if fix_data["xy"].shape[0] != 0:
            example_dict["X"] = fix_data["xy"][:, 0].tolist()
            example_dict["Y"] = fix_data["xy"][:, 1].tolist()
            example_dict["T_start"] = fix_data["t"][:, 0].tolist()
            example_dict["T_end"] = fix_data["t"][:, 1].tolist()

            example_dict["length"] = fix_data["t"].shape[0]
            all_fixation_length.append(example_dict["length"])
        else:
            except_list.append(fix_dir + "    " + subject)
            print(fix_dir + "    " + subject)
            break

        example_dict["subject_answer"] = consolidated_answer[subject][qid_value]
        example_dict["accuracy"] = consolidated_answer["accuracy"][qid_value]

        example_dict["split"] = "test"

        question = consolidated_answer["question"][qid_value]
        answer = consolidated_answer["answer"][qid_value]

        scene_graph_info = val_scene_graph[image_ids[qid_value]]

        example_dict["objects"] = scene_graph_info["objects"]

        test_list.append(example_dict)

all_fixation_length = np.array(all_fixation_length)

save_json_file = './data/AiR_fixations_train.json'
with open(save_json_file, 'w') as f:
    json.dump(train_list, f, indent=2)

save_json_file = './data/AiR_fixations_validation.json'
with open(save_json_file, 'w') as f:
    json.dump(validation_list, f, indent=2)

save_json_file = './data/AiR_fixations_test.json'
with open(save_json_file, 'w') as f:
    json.dump(test_list, f, indent=2)
