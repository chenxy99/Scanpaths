import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import scipy.stats

import time
import os
import argparse
from os.path import join
from tqdm import tqdm
import datetime
import json
import sys

# from dataset.dataset import AiR, AiR_evaluation
from dataset.dataset import COCO_Search18, COCO_Search18_evaluation, COCO_Search18_rl
# from models.baseline import baseline
# from models.baseline_egcb import baseline
# from models.baseline_performance import baseline
# from models.baseline_scene_graph import baseline
# from models.baseline_gt_project_labels import baseline
# from models.baseline_attention import baseline
from models.baseline_attention_multihead import baseline
from models.loss import CrossEntropyLoss, DurationSmoothL1Loss
from utils.checkpointing import CheckpointManager
from utils.recording import RecordManager
from utils.evaluation import human_evaluation, evaluation, evaluation_performance_related, human_evaluation_mismatch
from utils.logger import Logger
from visualization.vistools import show_sequential_action_map, show_saliency_map, show_image
from models.sampling import Sampling

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="validation", help="Selecting running mode (default: validation)")
parser.add_argument("--img_dir", type=str, default="./data/images", help="Directory to the image data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="./data/fixations", help="Directory to the raw fixation file")
parser.add_argument("--att_dir", type=str, default="./data/fixation_maps", help="Directory to the attention maps")
parser.add_argument("--detector_dir", type=str, default="./data/detectors", help="Directory to the saliency maps")
parser.add_argument("--bert_pretrained_dir", type=str, default="./data/question_embedding",
                    help="Directory to the bert pretrained hidden state")
# parser.add_argument("--anno_dir", type=str, default="../data/maps", help="Directory to the saliency maps")
parser.add_argument("--width", type=int, default=320, help="Width of input data")
parser.add_argument("--height", type=int, default=240, help="Height of input data")
parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--detector_threshold", type=float, default=0.8, help="threshold for the detector")
parser.add_argument("--gpu_ids", type=list, default=[0, 1], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="./assets/log_20201113_0125_supervised_save",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
parser.add_argument("--performance_related", type=bool, default=True, help="Consider the performance related or not")
parser.add_argument("--ablate_attention_info", type=bool, default=False, help="Ablate the attention information or not")
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([
                                transforms.Resize((args.height, args.width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def main():

    # load logger
    log_dir = args.evaluation_dir
    hparams_file = os.path.join(log_dir, "hparams.json")
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    log_file = os.path.join(log_dir, "log_validation.txt")
    predicts_file = os.path.join(log_dir, "validation_predicts.json")
    logger = Logger(log_file)

    logger.info("The args corresponding to validation process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    validation_dataset = COCO_Search18_evaluation(args.img_dir, args.fix_dir, args.detector_dir, None,
                                                  type="validation", split="split3", transform=transform,
                                                  detector_threshold=args.detector_threshold)

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=validation_dataset.collate_func
    )

    object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                   "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                   "toilet", "tv"]

    model = baseline(embed_size=512, bert_embed_size=768,
                     convLSTM_length=args.max_length, min_length=args.min_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    validation_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in validation_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(validation_checkpoint[key], strict=False)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # get the human baseline score
    human_metrics, human_metrics_std, gt_scores_of_each_images = human_evaluation(validation_loader)
    logger.info("The metrics for human performance are: ")
    for metrics_key in human_metrics.keys():
        for (key, value) in human_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
                        (metrics_key=metrics_key, key=key, value=value, std=human_metrics_std[metrics_key][key]))

    model.eval()
    repeat_num = args.eval_repeat_num
    x_granularity = float(args.width / args.map_width)
    y_granularity = float(args.height / args.map_height)
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    all_performances = []
    all_allocated_performances = []
    predict_results = []
    with tqdm(total=len(validation_loader) * repeat_num) as pbar_val:
        for i_batch, batch in enumerate(validation_loader):
            tmp = [batch["images"], batch["fix_vectors"], batch["attention_maps"], batch["tasks"],
                   batch["img_names"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            images, gt_fix_vectors, attention_maps, tasks, img_names = tmp

            N, C, H, W = images.shape

            if args.ablate_attention_info:
                attention_maps *= 0

            with torch.no_grad():
                predict = model(images, attention_maps, tasks)

            log_normal_mu = predict["log_normal_mu"]
            log_normal_sigma2 = predict["log_normal_sigma2"]
            all_actions_prob = predict["all_actions_prob"]

            for trial in range(repeat_num):
                all_gt_fix_vectors.extend(gt_fix_vectors)

                samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                prob_sample_actions = samples["selected_actions_probs"]
                durations = samples["durations"]
                sample_actions = samples["selected_actions"]
                sampling_random_predict_fix_vectors, _, _ = sampling.generate_scanpath(
                    images, prob_sample_actions, durations, sample_actions)
                all_predict_fix_vectors.extend(sampling_random_predict_fix_vectors)

                for index in range(N):
                    predict_result = dict()
                    one_sampling_random_predict_fix_vectors = sampling_random_predict_fix_vectors[index]
                    fix_vector_array = np.array(one_sampling_random_predict_fix_vectors.tolist())
                    predict_result["img_names"] = img_names[index]
                    predict_result["task"] = object_name[tasks[index]]
                    predict_result["repeat_id"] = trial + 1
                    predict_result["X"] = list(fix_vector_array[:, 0])
                    predict_result["Y"] = list(fix_vector_array[:, 1])
                    predict_result["T"] = list(fix_vector_array[:, 2] * 1000)
                    predict_result["length"] = len(predict_result["X"])
                    predict_results.append(predict_result)

                pbar_val.update(1)


    cur_metrics, cur_metrics_std, scores_of_each_images = evaluation(all_gt_fix_vectors, all_predict_fix_vectors)

    for index in range(len(predict_results)):
        predict_results[index]["scores"] = scores_of_each_images[index]
    with open(predicts_file, 'w') as f:
        json.dump(predict_results, f, indent=2)

    logger.info("The metrics for best model performance are: ")
    for metrics_key in cur_metrics.keys():
        for (metric_name, metric_value) in cur_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f} +- {std:.4f}".format
                        (metrics_key=metrics_key, metric_name=metric_name, metric_value=metric_value,
                         std=cur_metrics_std[metrics_key][metric_name]))


if __name__ == "__main__":
    main()
