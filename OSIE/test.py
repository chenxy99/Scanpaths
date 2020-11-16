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

from dataset.dataset import OSIE, OSIE_evaluation
# from models.baseline import baseline
# from models.baseline_egcb import baseline
# from models.baseline_attention import baseline
from models.baseline_attention_vgg import baseline
from models.loss import CrossEntropyLoss, DurationSmoothL1Loss
from utils.checkpointing import CheckpointManager
from utils.recording import RecordManager
from utils.evaluation import human_evaluation, evaluation
from utils.logger import Logger
from visualization.vistools import show_sequential_action_map, show_saliency_map, show_image
from models.sampling import Sampling

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="./data/stimuli", help="Directory to the image data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="./data/fixations_ior_roi", help="Directory to the raw fixation file")
parser.add_argument("--sal_dir", type=str, default="./data/fixation_maps", help="Directory to the saliency maps")
# parser.add_argument("--anno_dir", type=str, default="../data/maps", help="Directory to the saliency maps")
parser.add_argument("--width", type=int, default=320, help="Width of input data")
parser.add_argument("--height", type=int, default=240, help="Height of input data")
parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0, 1], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="./assets/vgg_backbone/log_20201114_1500_supervised_save/",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
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
    log_file = os.path.join(log_dir, "log_test.txt")
    predicts_file = os.path.join(log_dir, "test_predicts.json")
    logger = Logger(log_file)

    logger.info("The args corresponding to testing process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    test_dataset = OSIE_evaluation(args.img_dir, args.fix_dir, args.sal_dir, type="test", transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )

    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key])

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # get the human baseline score
    human_metrics, human_metrics_std, gt_scores_of_each_images = human_evaluation(test_loader)
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
    predict_results = []
    with tqdm(total=len(test_loader) * repeat_num) as pbar_test:
        for i_batch, batch in enumerate(test_loader):
            tmp = [batch["images"], batch["saliency_maps"], batch["fix_vectors"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            images, saliency_maps, gt_fix_vectors = tmp
            N, C, H, W = images.shape

            with torch.no_grad():
                predict = model(images)

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

                pbar_test.update(1)

    cur_metrics, cur_metrics_std, scores_of_each_images = evaluation(all_gt_fix_vectors, all_predict_fix_vectors)

    for index in range(len(predict_results)):
        predict_results[index]["scores"] = scores_of_each_images[index]
    with open(predicts_file, 'w') as f:
        json.dump(predict_results, f, indent=2)

    logger.info("The metrics for best model performance are: ")
    for metrics_key in cur_metrics.keys():
        for (key, value) in cur_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
                        (metrics_key=metrics_key, key=key, value=value, std=cur_metrics_std[metrics_key][key]))


if __name__ == "__main__":
    main()
