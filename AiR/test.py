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

from dataset.dataset import AiR, AiR_evaluation
from models.baseline_attention import baseline
from utils.evaluation import human_evaluation, evaluation_performance_related
from utils.logger import Logger
from models.sampling import Sampling

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="./data/stimuli", help="Directory to the image data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="./data/fixations", help="Directory to the raw fixation file")
parser.add_argument("--att_dir", type=str, default="./data/attention_reasoning", help="Directory to the attention maps")
parser.add_argument("--width", type=int, default=320, help="Width of input data")
parser.add_argument("--height", type=int, default=240, help="Height of input data")
parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="./assets/pretrained_model",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
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
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    log_file = os.path.join(log_dir, "log_test.txt")
    predicts_file = os.path.join(log_dir, "test_predicts.json")
    logger = Logger(log_file)

    logger.info("The args corresponding to testing process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    test_dataset = AiR_evaluation(args.img_dir, args.fix_dir, args.att_dir,
                                  type="test", transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )

    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length, map_width=args.map_width,
        map_height=args.map_height)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key], strict=False)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # get the human baseline score
    human_metrics, human_metrics_std, _ = human_evaluation(test_loader)
    logger.info("The metrics for human performance are: ")
    for category_key in human_metrics.keys():
        for metrics_key in human_metrics[category_key].keys():
            for (key, value) in human_metrics[category_key][metrics_key].items():
                logger.info("{category_key:12}: {metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
                            (category_key=category_key, metrics_key=metrics_key, key=key, value=value,
                             std=human_metrics_std[category_key][metrics_key][key]))
        logger.info("-" * 40)

    model.eval()
    repeat_num = args.eval_repeat_num
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    all_performances = []
    all_allocated_performances = []
    predict_results = []
    with tqdm(total=len(test_loader) * repeat_num) as pbar_test:
        for i_batch, batch in enumerate(test_loader):
            tmp = [batch["images"], batch["fix_vectors"], batch["performances"], batch["attention_maps"],
                   batch["question_ids"], batch["img_names"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            images, gt_fix_vectors, performances, attention_maps, question_ids, img_names = tmp
            N, C, H, W = images.shape

            if args.ablate_attention_info:
                attention_maps *= 0

            with torch.no_grad():
                predict = model(images, attention_maps)

            good_log_normal_mu = predict["good_log_normal_mu"]
            good_log_normal_sigma2 = predict["good_log_normal_sigma2"]
            good_all_actions_prob = predict["good_all_actions_prob"]
            poor_log_normal_mu = predict["poor_log_normal_mu"]
            poor_log_normal_sigma2 = predict["poor_log_normal_sigma2"]
            poor_all_actions_prob = predict["poor_all_actions_prob"]

            for trial in range(repeat_num):
                all_gt_fix_vectors.extend(gt_fix_vectors)
                all_performances.extend(performances)
                all_allocated_performances.extend([True] * N)

                samples = sampling.random_sample(good_all_actions_prob, good_log_normal_mu, good_log_normal_sigma2)
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
                    predict_result["qid"] = question_ids[index]
                    predict_result["repeat_id"] = trial + 1
                    predict_result["performance"] = True
                    predict_result["X"] = list(fix_vector_array[:, 0])
                    predict_result["Y"] = list(fix_vector_array[:, 1])
                    predict_result["T"] = list(fix_vector_array[:, 2] * 1000)
                    predict_result["length"] = len(predict_result["X"])
                    predict_results.append(predict_result)

                all_gt_fix_vectors.extend(gt_fix_vectors)
                all_performances.extend(performances)
                all_allocated_performances.extend([False] * N)

                samples = sampling.random_sample(poor_all_actions_prob, poor_log_normal_mu, poor_log_normal_sigma2)
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
                    predict_result["qid"] = question_ids[index]
                    predict_result["repeat_id"] = trial + 1
                    predict_result["performance"] = False
                    predict_result["X"] = list(fix_vector_array[:, 0])
                    predict_result["Y"] = list(fix_vector_array[:, 1])
                    predict_result["T"] = list(fix_vector_array[:, 2] * 1000)
                    predict_result["length"] = len(predict_result["X"])
                    predict_results.append(predict_result)

                pbar_test.update(1)


    cur_metrics, cur_metrics_std, _ = evaluation_performance_related(all_gt_fix_vectors, all_predict_fix_vectors,
                                                                  all_performances, all_allocated_performances)

    with open(predicts_file, 'w') as f:
        json.dump(predict_results, f, indent=2)

    logger.info("The metrics for best model performance are: ")
    for category_key in cur_metrics.keys():
        for metrics_key in cur_metrics[category_key].keys():
            for (metric_name, metric_value) in cur_metrics[category_key][metrics_key].items():
                logger.info(
                    "{category_key:12}: {metrics_key:10}-{metric_name:15}: {metric_value:.4f} +- {std:.4f}".format
                    (category_key=category_key, metrics_key=metrics_key, metric_name=metric_name,
                     metric_value=metric_value,
                     std=cur_metrics_std[category_key][metrics_key][metric_name]))
        logger.info("-" * 40)


if __name__ == "__main__":
    main()
