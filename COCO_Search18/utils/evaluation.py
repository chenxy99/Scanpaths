import torch
import numpy as np
import scipy.stats
import copy

from tqdm import tqdm
import multimatch_gaze as multimatch
from utils.evaltools.scanmatch import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity

def human_evaluation(dataloader):
    collect_multimatch_rlts = []
    collect_scanmatch_with_duration_rlts = []
    collect_scanmatch_without_duration_rlts = []
    collect_SED_rlts = []
    collect_STDE_rlts = []
    scores_of_each_images = []
    start_index_of_vector = [0]
    start_index_of_image = [0]

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    gt_image_name = list()
    with tqdm(total=len(dataloader)) as pbar:
        for i_batch, batch in enumerate(dataloader):
            batch_fix_vectors = batch["fix_vectors"]
            gt_image_name.extend(batch["img_names"])
            for fix_vectors in batch_fix_vectors:
                scores_of_given_image = []
                for index_1 in range(len(fix_vectors)):
                    fix_vector_1 = fix_vectors[index_1]
                    for index_2 in range(0, len(fix_vectors)):
                        if index_2 == index_1:
                            continue
                        fix_vector_2 = fix_vectors[index_2]
                        # calculate multimatch
                        rlt = multimatch.docomparison(fix_vector_1, fix_vector_2, screensize=[320, 240])
                        collect_multimatch_rlts.append(rlt)
                        scores_of_given_image_with_gt = list(copy.deepcopy(rlt))

                        # perform scanmatch
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array([list(_) for _ in list(fix_vector_1)])
                        np_fix_vector_2 = np.array([list(_) for _ in list(fix_vector_2)])
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score, align, f) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                        collect_scanmatch_with_duration_rlts.append(score)
                        scores_of_given_image_with_gt.append(score)
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score, align, f) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                        collect_scanmatch_without_duration_rlts.append(score)
                        scores_of_given_image_with_gt.append(score)

                        # perfrom SED
                        sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                        collect_SED_rlts.append(sed)
                        scores_of_given_image_with_gt.append(sed)

                        # perfrom STDE
                        stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                        collect_STDE_rlts.append(stde)
                        scores_of_given_image_with_gt.append(stde)
                        scores_of_given_image.append(scores_of_given_image_with_gt)
                    start_index_of_vector.append(start_index_of_vector[-1] + len(fix_vectors) - 1)
                start_index_of_image.append(start_index_of_image[-1] + len(fix_vectors))
                scores_of_each_images.append(list(np.array(scores_of_given_image).mean(axis=0)))
            pbar.update(1)

    collect_multimatch_rlts = np.array(collect_multimatch_rlts)
    collect_multimatch_rlts = collect_multimatch_rlts[np.isnan(collect_multimatch_rlts.sum(axis=1)) == False]
    multimatch_metric_mean = np.mean(collect_multimatch_rlts, axis=0)
    multimatch_metric_std = np.std(collect_multimatch_rlts, axis=0)

    scanmatch_with_duration_metric_mean = np.mean(collect_scanmatch_with_duration_rlts)
    scanmatch_with_duration_metric_std = np.std(collect_scanmatch_with_duration_rlts)
    scanmatch_without_duration_metric_mean = np.mean(collect_scanmatch_without_duration_rlts)
    scanmatch_without_duration_metric_std = np.std(collect_scanmatch_without_duration_rlts)

    sample_collect_SED_rlts = []
    for sample_index in range(1, len(start_index_of_image)):
        start_idx = start_index_of_image[sample_index - 1]
        end_idx = start_index_of_image[sample_index]
        sub_vector = start_index_of_vector[start_idx:end_idx+1]
        sample_collect_SED_rlt = []
        for ii in range(1, len(sub_vector)):
            start_vec_idx = sub_vector[ii - 1]
            end_vec_idx = sub_vector[ii]
            sample_collect_SED_rlt.append(collect_SED_rlts[start_vec_idx:end_vec_idx])
        sample_collect_SED_rlts.append(np.array(sample_collect_SED_rlt))

    sample_collect_STDE_rlts = []
    for sample_index in range(1, len(start_index_of_image)):
        start_idx = start_index_of_image[sample_index - 1]
        end_idx = start_index_of_image[sample_index]
        sub_vector = start_index_of_vector[start_idx:end_idx + 1]
        sample_collect_STDE_rlt = []
        for ii in range(1, len(sub_vector)):
            start_vec_idx = sub_vector[ii - 1]
            end_vec_idx = sub_vector[ii]
            sample_collect_STDE_rlt.append(collect_STDE_rlts[start_vec_idx:end_vec_idx])
        sample_collect_STDE_rlts.append(np.array(sample_collect_STDE_rlt))

    # SED_metrics_rlts = np.array(collect_SED_rlts)
    # STDE_metrics_rlts = np.array(collect_STDE_rlts)
    # SED_metrics_rlts = SED_metrics_rlts.reshape(-1, len(fix_vectors) - 1)
    # STDE_metrics_rlts = STDE_metrics_rlts.reshape(-1, len(fix_vectors) - 1)

    SED_metrics_mean = np.array(collect_SED_rlts).mean()
    SED_metrics_std = np.array(collect_SED_rlts).std()
    STDE_metrics_mean = np.array(collect_STDE_rlts).mean()
    STDE_metrics_std = np.array(collect_STDE_rlts).std()

    SED_metrics_best_rlts = [data_val.min() for data_val in sample_collect_SED_rlts]
    STDE_metrics_best_rlts = [data_val.max() for data_val in sample_collect_STDE_rlts]
    SED_best_metrics= np.array(SED_metrics_best_rlts)
    STDE_best_metrics = np.array(STDE_metrics_best_rlts)
    SED_best_metrics_mean = SED_best_metrics.mean()
    SED_best_metrics_std = SED_best_metrics.std()
    STDE_best_metrics_mean = STDE_best_metrics.mean()
    STDE_best_metrics_std = STDE_best_metrics.std()


    human_metrics = dict()
    human_metrics_std = dict()

    multimatch_human_metrics = dict()
    multimatch_human_metrics["vector"] = multimatch_metric_mean[0]
    multimatch_human_metrics["direction"] = multimatch_metric_mean[1]
    multimatch_human_metrics["length"] = multimatch_metric_mean[2]
    multimatch_human_metrics["position"] = multimatch_metric_mean[3]
    multimatch_human_metrics["duration"] = multimatch_metric_mean[4]
    human_metrics["MultiMatch"] = multimatch_human_metrics

    scanmatch_human_metrics = dict()
    scanmatch_human_metrics["w/o duration"] = scanmatch_without_duration_metric_mean
    scanmatch_human_metrics["with duration"] = scanmatch_with_duration_metric_mean
    human_metrics["ScanMatch"] = scanmatch_human_metrics

    multimatch_human_metrics_std = dict()
    multimatch_human_metrics_std["vector"] = multimatch_metric_std[0]
    multimatch_human_metrics_std["direction"] = multimatch_metric_std[1]
    multimatch_human_metrics_std["length"] = multimatch_metric_std[2]
    multimatch_human_metrics_std["position"] = multimatch_metric_std[3]
    multimatch_human_metrics_std["duration"] = multimatch_metric_std[4]
    human_metrics_std["MultiMatch"] = multimatch_human_metrics_std

    scanmatch_human_metrics_std = dict()
    scanmatch_human_metrics_std["w/o duration"] = scanmatch_without_duration_metric_std
    scanmatch_human_metrics_std["with duration"] = scanmatch_with_duration_metric_std
    human_metrics_std["ScanMatch"] = scanmatch_human_metrics_std

    VAME_human_metrics = dict()
    VAME_human_metrics["SED"] = SED_metrics_mean
    VAME_human_metrics["STDE"] = STDE_metrics_mean
    VAME_human_metrics["SED_best"] = SED_best_metrics_mean
    VAME_human_metrics["STDE_best"] = STDE_best_metrics_mean
    human_metrics["VAME"] = VAME_human_metrics

    VAME_human_metrics_std = dict()
    VAME_human_metrics_std["SED"] = SED_metrics_std
    VAME_human_metrics_std["STDE"] = STDE_metrics_std
    VAME_human_metrics_std["SED_best"] = SED_best_metrics_std
    VAME_human_metrics_std["STDE_best"] = STDE_best_metrics_std
    human_metrics_std["VAME"] = VAME_human_metrics_std

    scores_of_each_images_dict = dict()
    for name, score in zip(gt_image_name, scores_of_each_images):
        scores_of_each_images_dict[name] = score
    return human_metrics, human_metrics_std, scores_of_each_images_dict

def evaluation(gt_fix_vectors, predict_fix_vectors, is_eliminating_nan=True):
    collect_multimatch_rlts = []
    collect_scanmatch_with_duration_rlts = []
    collect_scanmatch_without_duration_rlts = []
    collect_SED_rlts = []
    collect_STDE_rlts = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)

    scores_of_each_images = []
    with tqdm(total=len(gt_fix_vectors)) as pbar:
        for index in range(len(gt_fix_vectors)):
            gt_fix_vector = gt_fix_vectors[index]
            predict_fix_vector = predict_fix_vectors[index]
            scores_of_given_image = []
            for inner_index in range(len(gt_fix_vector)):
                inner_gt_fix_vector = gt_fix_vector[inner_index]
                # calculate multimatch
                rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])
                collect_multimatch_rlts.append(rlt)
                scores_of_given_image_with_gt = list(copy.deepcopy(rlt))

                # perform scanmatch
                # we need to transform the scale of time from s to ms
                # with duration
                np_fix_vector_1 = np.array([list(_) for _ in list(inner_gt_fix_vector)])
                np_fix_vector_2 = np.array([list(_) for _ in list(predict_fix_vector)])
                np_fix_vector_1[:, -1] *= 1000
                np_fix_vector_2[:, -1] *= 1000
                sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                (score, align, f) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                collect_scanmatch_with_duration_rlts.append(score)
                scores_of_given_image_with_gt.append(score)
                # without duration
                sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                (score, align, f) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                collect_scanmatch_without_duration_rlts.append(score)
                scores_of_given_image_with_gt.append(score)

                # perfrom SED
                sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                collect_SED_rlts.append(sed)
                scores_of_given_image_with_gt.append(sed)

                # perfrom STDE
                stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                collect_STDE_rlts.append(stde)
                scores_of_given_image_with_gt.append(stde)

                scores_of_given_image.append(scores_of_given_image_with_gt)

            scores_of_each_images.append(list(np.array(scores_of_given_image).mean(axis=0)))
            pbar.update(1)

    collect_multimatch_rlts = np.array(collect_multimatch_rlts)
    if is_eliminating_nan:
        collect_multimatch_rlts = collect_multimatch_rlts[np.isnan(collect_multimatch_rlts.sum(axis=1)) == False]
    multimatch_metric_mean = np.mean(collect_multimatch_rlts, axis=0)
    multimatch_metric_std = np.std(collect_multimatch_rlts, axis=0)

    scanmatch_with_duration_metric_mean = np.mean(collect_scanmatch_with_duration_rlts)
    scanmatch_with_duration_metric_std = np.std(collect_scanmatch_with_duration_rlts)
    scanmatch_without_duration_metric_mean = np.mean(collect_scanmatch_without_duration_rlts)
    scanmatch_without_duration_metric_std = np.std(collect_scanmatch_without_duration_rlts)

    SED_metrics_rlts = np.array(collect_SED_rlts)
    STDE_metrics_rlts = np.array(collect_STDE_rlts)
    SED_metrics_rlts = SED_metrics_rlts.reshape(-1, len(gt_fix_vector))
    STDE_metrics_rlts = STDE_metrics_rlts.reshape(-1, len(gt_fix_vector))

    SED_metrics_mean = SED_metrics_rlts.mean()
    SED_metrics_std = SED_metrics_rlts.std()
    STDE_metrics_mean = STDE_metrics_rlts.mean()
    STDE_metrics_std = STDE_metrics_rlts.std()

    SED_best_metrics= SED_metrics_rlts.min(-1)
    STDE_best_metrics = STDE_metrics_rlts.max(-1)
    SED_best_metrics_mean = SED_best_metrics.mean()
    SED_best_metrics_std = SED_best_metrics.std()
    STDE_best_metrics_mean = STDE_best_metrics.mean()
    STDE_best_metrics_std = STDE_best_metrics.std()

    cur_metrics = dict()
    cur_metrics_std = dict()

    multimatch_cur_metrics = dict()
    multimatch_cur_metrics["vector"] = multimatch_metric_mean[0]
    multimatch_cur_metrics["direction"] = multimatch_metric_mean[1]
    multimatch_cur_metrics["length"] = multimatch_metric_mean[2]
    multimatch_cur_metrics["position"] = multimatch_metric_mean[3]
    multimatch_cur_metrics["duration"] = multimatch_metric_mean[4]
    cur_metrics["MultiMatch"] = multimatch_cur_metrics

    scanmatch_cur_metrics = dict()
    scanmatch_cur_metrics["w/o duration"] = scanmatch_without_duration_metric_mean
    scanmatch_cur_metrics["with duration"] = scanmatch_with_duration_metric_mean
    cur_metrics["ScanMatch"] = scanmatch_cur_metrics

    multimatch_cur_metrics_std = dict()
    multimatch_cur_metrics_std["vector"] = multimatch_metric_std[0]
    multimatch_cur_metrics_std["direction"] = multimatch_metric_std[1]
    multimatch_cur_metrics_std["length"] = multimatch_metric_std[2]
    multimatch_cur_metrics_std["position"] = multimatch_metric_std[3]
    multimatch_cur_metrics_std["duration"] = multimatch_metric_std[4]
    cur_metrics_std["MultiMatch"] = multimatch_cur_metrics_std

    scanmatch_cur_metrics_std = dict()
    scanmatch_cur_metrics_std["w/o duration"] = scanmatch_without_duration_metric_std
    scanmatch_cur_metrics_std["with duration"] = scanmatch_with_duration_metric_std
    cur_metrics_std["ScanMatch"] = scanmatch_cur_metrics_std

    VAME_cur_metrics = dict()
    VAME_cur_metrics["SED"] = SED_metrics_mean
    VAME_cur_metrics["STDE"] = STDE_metrics_mean
    VAME_cur_metrics["SED_best"] = SED_best_metrics_mean
    VAME_cur_metrics["STDE_best"] = STDE_best_metrics_mean
    cur_metrics["VAME"] = VAME_cur_metrics

    VAME_cur_metrics_std = dict()
    VAME_cur_metrics_std["SED"] = SED_metrics_std
    VAME_cur_metrics_std["STDE"] = STDE_metrics_std
    VAME_cur_metrics_std["SED_best"] = SED_best_metrics_std
    VAME_cur_metrics_std["STDE_best"] = STDE_best_metrics_std
    cur_metrics_std["VAME"] = VAME_cur_metrics_std

    return cur_metrics, cur_metrics_std, scores_of_each_images

def pairs_eval_scanmatch(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration,
                         is_eliminating_nan=True):
    # [0]: scanmatch_with_duration; [1]: scanmatch_without_duration
    pairs_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_rlts = []
        for inner_index in range(len(gt_fix_vector)):
            inner_gt_fix_vector = gt_fix_vector[inner_index]

            # perform scanmatch
            # we need to transform the scale of time from s to ms
            # with duration
            np_fix_vector_1 = np.array([list(_) for _ in list(inner_gt_fix_vector)])
            np_fix_vector_2 = np.array([list(_) for _ in list(predict_fix_vector)])
            np_fix_vector_1[:, -1] *= 1000
            np_fix_vector_2[:, -1] *= 1000

            sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
            sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
            (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
            # without duration
            sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
            sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
            (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)

            rlt = [score_wod, score_wd]
            collect_rlts.append(rlt)
        collect_rlts = np.array(collect_rlts)
        if is_eliminating_nan:
            collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
        if collect_rlts.shape[0] != 0:
            metric_value = np.sum(collect_rlts, axis=0) / len(gt_fix_vector)
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_summary_metric.append(metric_value)

    return np.array(pairs_summary_metric)
