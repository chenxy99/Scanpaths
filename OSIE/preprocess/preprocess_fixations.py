import scipy.io as sio
import numpy as np
import json

# We use the same test set as paper
# "Visual Scanpath Prediction using IOR-ROI Recurrent Mixture Density Network" provided
test_image_names_list = ["1009.jpg", "1017.jpg", "1049.jpg", "1056.jpg", "1062.jpg", "1086.jpg", "1087.jpg",
                         "1099.jpg", "1108.jpg", "1114.jpg", "1116.jpg", "1117.jpg", "1127.jpg", "1130.jpg",
                         "1131.jpg", "1136.jpg", "1140.jpg", "1152.jpg", "1192.jpg", "1220.jpg", "1225.jpg",
                         "1226.jpg", "1252.jpg", "1255.jpg", "1269.jpg", "1295.jpg", "1307.jpg", "1360.jpg",
                         "1369.jpg", "1372.jpg", "1394.jpg", "1397.jpg", "1405.jpg", "1420.jpg", "1423.jpg",
                         "1433.jpg", "1441.jpg", "1478.jpg", "1480.jpg", "1481.jpg", "1489.jpg", "1490.jpg",
                         "1493.jpg", "1502.jpg", "1509.jpg", "1523.jpg", "1528.jpg", "1530.jpg", "1549.jpg",
                         "1555.jpg", "1558.jpg", "1567.jpg", "1576.jpg", "1581.jpg", "1595.jpg", "1596.jpg",
                         "1605.jpg", "1609.jpg", "1615.jpg", "1616.jpg", "1618.jpg", "1622.jpg", "1628.jpg",
                         "1637.jpg", "1640.jpg", "1657.jpg", "1663.jpg", "1677.jpg", "1682.jpg", "1699.jpg", ]


mat_file = '../preprocess_data/eye/fixations.mat'
data = sio.loadmat(mat_file)
fixations = data['fixations']

np.random.seed(0)
length_scanpath = []
duration_scanpath = []

trainval_image_names_list = list()
for example in fixations:
    example_value = example[0][0][0]
    if example_value[0].item() in test_image_names_list:
        pass
    else:
        trainval_image_names_list.append(example_value[0].item())
np.random.shuffle(trainval_image_names_list)

trainval_len = len(trainval_image_names_list)
train_image_names_list = trainval_image_names_list[: int(trainval_len * 8./9.)]
val_image_names_list = trainval_image_names_list[int(trainval_len * 8./9.):]

train_list = list()
for example in fixations:
    example_value = example[0][0][0]
    if example_value[0].item() in train_image_names_list:
        detail_data = example_value[1]
        for index in range(len(detail_data)):
            example_dict = dict()
            example_dict['name'] = example_value[0].item()
            example_dict['subject'] = index + 1
            example_dict['X'] = detail_data[index][0][0][0][0].squeeze(0).tolist()
            example_dict['Y'] = detail_data[index][0][0][0][1].squeeze(0).tolist()
            example_dict['T'] = detail_data[index][0][0][0][2].squeeze(0).tolist()
            example_dict['length'] = detail_data[index][0][0][0][0].squeeze(0).shape[0]
            example_dict['split'] = 'train'
            train_list.append(example_dict)
            length_scanpath.append(len(example_dict['T']))
            duration_scanpath.extend(example_dict['T'])

val_list = list()
for example in fixations:
    example_value = example[0][0][0]
    if example_value[0].item() in val_image_names_list:
        detail_data = example_value[1]
        for index in range(len(detail_data)):
            example_dict = dict()
            example_dict['name'] = example_value[0].item()
            example_dict['subject'] = index + 1
            example_dict['X'] = detail_data[index][0][0][0][0].squeeze(0).tolist()
            example_dict['Y'] = detail_data[index][0][0][0][1].squeeze(0).tolist()
            example_dict['T'] = detail_data[index][0][0][0][2].squeeze(0).tolist()
            example_dict['length'] = detail_data[index][0][0][0][0].squeeze(0).shape[0]
            example_dict['split'] = 'validation'
            val_list.append(example_dict)
            length_scanpath.append(len(example_dict['T']))
            duration_scanpath.extend(example_dict['T'])


test_list = list()
for example in fixations:
    example_value = example[0][0][0]
    if example_value[0].item() in test_image_names_list:
        detail_data = example_value[1]
        for index in range(len(detail_data)):
            example_dict = dict()
            example_dict['name'] = example_value[0].item()
            example_dict['subject'] = index + 1
            example_dict['X'] = detail_data[index][0][0][0][0].squeeze(0).tolist()
            example_dict['Y'] = detail_data[index][0][0][0][1].squeeze(0).tolist()
            example_dict['T'] = detail_data[index][0][0][0][2].squeeze(0).tolist()
            example_dict['length'] = detail_data[index][0][0][0][0].squeeze(0).shape[0]
            example_dict['split'] = 'test'
            test_list.append(example_dict)
            length_scanpath.append(len(example_dict['T']))
            duration_scanpath.extend(example_dict['T'])


save_json_file = '../data/fixations/osie_fixations_train.json'
with open(save_json_file, 'w') as f:
    json.dump(train_list, f, indent=2)

save_json_file = '../data/fixations/osie_fixations_validation.json'
with open(save_json_file, 'w') as f:
    json.dump(val_list, f, indent=2)

save_json_file = '../data/fixations/osie_fixations_test.json'
with open(save_json_file, 'w') as f:
    json.dump(test_list, f, indent=2)
