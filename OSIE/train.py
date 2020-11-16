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

from dataset.dataset import OSIE, OSIE_evaluation, OSIE_rl
# from models.baseline import baseline
# from models.baseline_egcb import baseline
# from models.baseline_attention import baseline
from models.baseline_attention_vgg import baseline
from models.loss import CrossEntropyLoss, DurationSmoothL1Loss, MLPRayleighDistribution, MLPLogNormalDistribution, \
    LogAction, LogDuration, NSS, CC, KLD
from utils.checkpointing import CheckpointManager
from utils.recording import RecordManager
from utils.evaluation import human_evaluation, evaluation, pairs_multimatch_eval, pairs_eval, pairs_VAME_eval, \
    pairs_scanmatch_eval
from utils.logger import Logger
from opts import parse_opt
from utils.evaltools.scanmatch import ScanMatch
from visualization.vistools import show_sequential_action_map, show_saliency_map, show_image
from models.sampling import Sampling

args = parse_opt()

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
    # setup logger
    if args.resume_dir == "":
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        log_dir = os.path.join(args.log_root, "log_" + date)
    else:
        log_dir = args.resume_dir
    hparams_file = os.path.join(log_dir, "hparams.json")
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if args.resume_dir == "":
        # write hparams
        with open(hparams_file, "w") as f:
            json.dump(args.__dict__, f, indent=2)
    log_file = os.path.join(log_dir, "log_train.txt")
    logger = Logger(log_file)
    # logger.info(args)
    logger.info("The args corresponding to training process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))
    # else:
    #     # read hparams
    #     with open(hparams_file, 'r') as f:
    #         args.__dict__ = json.load(f)

    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE VOCABULARY, DATALOADER, MODEL, OPTIMIZER
    # --------------------------------------------------------------------------------------------

    train_dataset = OSIE(args.img_dir, args.fix_dir, args.sal_dir,
                         blur_sigma=args.blur_sigma, type="train", transform=transform)
    train_dataset_rl = OSIE_rl(args.img_dir, args.fix_dir, type="train", transform=transform)
    validation_dataset = OSIE_evaluation(args.img_dir, args.fix_dir, args.sal_dir,
                                         type="validation", transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_func
    )
    train_rl_loader = DataLoader(
        dataset=train_dataset_rl,
        batch_size=args.batch // 4,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset_rl.collate_func
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=validation_dataset.collate_func
    )

    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length)

    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=args.weight_decay)

    # --------------------------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # --------------------------------------------------------------------------------------------

    # Tensorboard summary writer for logging losses and metrics.
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    # Record manager for writing and loading the best metrics and theirs corresponding epoch
    record_manager = RecordManager(log_dir)
    if args.resume_dir == '':
        record_manager.init_record()
    else:
        record_manager.load()

    start_epoch = record_manager.get_epoch()
    iteration = record_manager.get_iteration()
    best_metric = record_manager.get_best_metric()


    # Checkpoint manager to serialize checkpoints periodically while training and keep track of
    # best performing checkpoint.
    checkpoint_manager = CheckpointManager(model, optimizer, checkpoints_dir, mode="max", best_metric=best_metric)

    # Load checkpoint to resume training from there if specified.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    # saved checkpoints if you intend to continue training.
    if args.resume_dir != "":
        training_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint.pth"))
        for key in training_checkpoint:
            if key == "optimizer":
                optimizer.load_state_dict(training_checkpoint[key])
            else:
                model.load_state_dict(training_checkpoint[key])

    # lr_scheduler = optim.lr_scheduler.LambdaLR \
    #     (optimizer, lr_lambda=lambda iteration: 1 - iteration / (len(train_loader) * args.epoch), last_epoch=iteration)

    def lr_lambda(iteration):
        if iteration <= len(train_loader) * args.warmup_epoch:
            return iteration / (len(train_loader) * args.warmup_epoch)
        elif iteration <= len(train_loader) * args.start_rl_epoch:
            return 1 - (iteration - len(train_loader) * args.warmup_epoch) / \
                   (len(train_loader) * (args.start_rl_epoch - args.warmup_epoch))
        else:
            return args.rl_lr_initial_decay * (1 - (iteration - (len(train_loader) * args.start_rl_epoch)) /
                                               (len(train_rl_loader) * (args.epoch - args.start_rl_epoch)))
            pass

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=iteration)



    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    def train(iteration, epoch):
        # traditional training stage
        if epoch < args.start_rl_epoch:
            model.train()
            for i_batch, batch in enumerate(train_loader):
                tmp = [batch["images"], batch["saliency_maps"], batch["fixation_maps"],
                       batch["scanpaths"], batch["durations"],
                       batch["action_masks"], batch["duration_masks"]]
                tmp = [_ if _ is None else _.cuda() for _ in tmp]
                images, saliency_maps, fixation_maps, scanpaths, durations, action_masks, duration_masks = tmp

                optimizer.zero_grad()
                # predicts = model(images, scanpaths, durations, duration_masks)
                predicts = model(images)

                loss_actions = CrossEntropyLoss(predicts["actions"], scanpaths, action_masks)
                loss_duration = MLPLogNormalDistribution(predicts["log_normal_mu"], predicts["log_normal_sigma2"],
                                                         durations, duration_masks)
                # loss_NSS = NSS(predicts["saliency_map_predictions"], fixation_maps)
                # loss_CC = CC(predicts["saliency_map_predictions"], saliency_maps)
                # loss_KLD = KLD(predicts["saliency_map_predictions"], saliency_maps)
                # loss = loss_actions + args.lambda_1 * loss_duration + args.lambda_2 * loss_NSS \
                #        + args.lambda_3 * loss_CC + args.lambda_4 * loss_KLD
                loss = loss_actions + args.lambda_1 * loss_duration

                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                iteration += 1
                lr_scheduler.step()
                pbar.update(1)
                # Log loss and learning rate to tensorboard.
                tensorboard_writer.add_scalar("loss/loss", loss, iteration)
                tensorboard_writer.add_scalar("loss/loss_actions", loss_actions, iteration)
                tensorboard_writer.add_scalar("loss/loss_duration", loss_duration, iteration)
                # tensorboard_writer.add_scalar("loss/loss_NSS", loss_NSS, iteration)
                # tensorboard_writer.add_scalar("loss/loss_CC", loss_CC, iteration)
                # tensorboard_writer.add_scalar("loss/loss_KLD", loss_KLD, iteration)
                tensorboard_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], iteration)
        # reinforcement learning stage
        else:
            model.eval()
            # create a ScanMatch object
            ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50,
                                              Threshold=3.5)
            ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)
            x_granularity = float(args.width / args.map_width)
            y_granularity = float(args.height / args.map_height)
            for i_batch, batch in enumerate(train_rl_loader):
                tmp = [batch["images"], batch["fix_vectors"]]
                tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
                images, gt_fix_vectors = tmp
                N, C, H, W = images.shape

                optimizer.zero_grad()

                metrics_reward_batch = []
                neg_log_actions_batch = []
                neg_log_durations_batch = []

                predict = model(images)

                log_normal_mu = predict["log_normal_mu"]
                log_normal_sigma2 = predict["log_normal_sigma2"]
                all_actions_prob = predict["all_actions_prob"]

                trial = 0
                while True:
                    if trial >= args.rl_sample_number:
                        break

                    samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                    prob_sample_actions = samples["selected_actions_probs"]
                    durations = samples["durations"]
                    sample_actions = samples["selected_actions"]
                    random_predict_fix_vectors, action_masks, duration_masks = sampling.generate_scanpath(
                        images, prob_sample_actions, durations, sample_actions)
                    t = durations.data.clone()
                    metrics_reward = pairs_eval(gt_fix_vectors, random_predict_fix_vectors,
                                                          ScanMatchwithDuration, ScanMatchwithoutDuration)

                    if np.any(np.isnan(metrics_reward)):
                        continue
                    else:
                        trial += 1
                        metrics_reward = torch.tensor(metrics_reward, dtype=torch.float32).to(images.get_device())
                        neg_log_actions = - LogAction(prob_sample_actions, action_masks)
                        neg_log_durations = - LogDuration(t, log_normal_mu, log_normal_sigma2, duration_masks)
                        metrics_reward_batch.append(metrics_reward.unsqueeze(0))
                        neg_log_actions_batch.append(neg_log_actions.unsqueeze(0))
                        neg_log_durations_batch.append(neg_log_durations.unsqueeze(0))

                neg_log_actions_tensor = torch.cat(neg_log_actions_batch, dim=0)
                neg_log_durations_tensor = torch.cat(neg_log_durations_batch, dim=0)
                # use the mean as reward
                # metrics_reward_tensor = torch.cat(metrics_reward_batch, dim=0)
                # baseline_reward_tensor = metrics_reward_tensor.mean(0, keepdim=True)
                # loss_actions = (neg_log_actions_tensor * (metrics_reward_tensor - baseline_reward_tensor).sum(-1)).sum()
                # loss_duration = (
                #             neg_log_durations_tensor * (metrics_reward_tensor - baseline_reward_tensor).sum(-1)).sum()
                # loss = loss_actions + loss_duration
                # use the hmean as reward
                metrics_reward_tensor = torch.cat(metrics_reward_batch, dim=0)
                metrics_reward_hmean = scipy.stats.hmean(metrics_reward_tensor[:, :, 5:7].cpu(), axis=-1)
                metrics_reward_hmean_tensor = torch.tensor(metrics_reward_hmean).to(metrics_reward_tensor.get_device())
                baseline_reward_hmean_tensor = metrics_reward_hmean_tensor.mean(0, keepdim=True)

                loss_actions = (neg_log_actions_tensor * (metrics_reward_hmean_tensor - baseline_reward_hmean_tensor)).sum()
                loss_duration = (neg_log_durations_tensor * (metrics_reward_hmean_tensor - baseline_reward_hmean_tensor)).sum()
                loss = loss_actions + loss_duration

                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                iteration += 1
                lr_scheduler.step()
                pbar.update(1)
                # Log loss and learning rate to tensorboard.
                multimatch_metric_names = ["vector", "direction", "length", "position", "duration",
                                           "w/o duration", "w/ duration", "SED mean",
                                           "STDE mean", "SED best", "STDE best"]
                multimatch_metrics_reward = metrics_reward_tensor.mean(0).mean(0)
                tensorboard_writer.add_scalar("rl_loss", loss, iteration)
                tensorboard_writer.add_scalar("reward_hmean", metrics_reward_hmean.mean(), iteration)
                tensorboard_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], iteration)
                for metric_index in range(len(multimatch_metric_names)):
                    tensorboard_writer.add_scalar(
                        "metrics_for_reward/{metric_name}".format(metric_name=multimatch_metric_names[metric_index]),
                        multimatch_metrics_reward[metric_index], iteration
                    )

        return iteration

    def validation(iteration):
        model.eval()
        repeat_num = args.eval_repeat_num
        x_granularity = float(args.width / args.map_width)
        y_granularity = float(args.height / args.map_height)
        all_gt_fix_vectors = []
        all_predict_fix_vectors = []
        with tqdm(total=len(validation_loader) * repeat_num) as pbar_val:
            for i_batch, batch in enumerate(validation_loader):
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

                    pbar_val.update(1)

        cur_metrics, cur_metrics_std, _ = evaluation(all_gt_fix_vectors, all_predict_fix_vectors)

        # Print and log all evaluation metrics to tensorboard.
        logger.info("Evaluation metrics after iteration {iteration}:".format(iteration=iteration))
        for metrics_key in cur_metrics.keys():
            for (metric_name, metric_value) in cur_metrics[metrics_key].items():
                tensorboard_writer.add_scalar(
                    "metrics/{metrics_key}-{metric_name}".format(metrics_key=metrics_key, metric_name=metric_name),
                    metric_value, iteration
                )
                logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f} +- {std:.4f}".format
                            (metrics_key=metrics_key, metric_name=metric_name, metric_value=metric_value,
                             std=cur_metrics_std[metrics_key][metric_name]))

        return cur_metrics


    # get the human baseline score
    # human_metrics, human_metrics_std = human_evaluation(validation_loader)
    # logger.info("The metrics for human performance are: ")
    # for metrics_key in human_metrics.keys():
    #     for (key, value) in human_metrics[metrics_key].items():
    #         logger.info("{metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
    #                     (metrics_key=metrics_key, key=key, value=value, std=human_metrics_std[metrics_key][key]))

    tqdm_total = len(train_loader) * args.start_rl_epoch + len(train_rl_loader) * (args.epoch - args.start_rl_epoch)
    with tqdm(total=tqdm_total, initial=iteration + 1) as pbar:
        for epoch in range(start_epoch + 1, args.epoch):
            iteration = train(iteration, epoch)
            cur_metrics = validation(iteration)
            cur_metric = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))

            # Log current metric to tensorboard.
            tensorboard_writer.add_scalar("current metric", float(cur_metric), iteration)
            logger.info("{key:10}: {value:.4f}".format(key="current metric", value=float(cur_metric)))

            # save
            checkpoint_manager.step(float(cur_metric))
            best_metric = checkpoint_manager.get_best_metric()
            record_manager.save(epoch, iteration, best_metric)

            # check  whether to save the final supervised training file
            if args.supervised_save and epoch == args.start_rl_epoch - 1:
                cmd = 'cp -r ' + log_dir + ' ' + log_dir + '_supervised_save'
                os.system(cmd)


if __name__ == "__main__":
    main()
