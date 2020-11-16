import torch
import numpy as np
import scipy.stats
import copy

from tqdm import tqdm
import multimatch_gaze as multimatch
from utils.evaltools.scanmatch import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity

def human_evaluation(dataloader):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []
    good_scores_of_each_images = []
    poor_scores_of_each_images = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    gt_qid_name = list()
    with tqdm(total=len(dataloader)) as pbar:
        for i_batch, batch in enumerate(dataloader):
            batch_fix_vectors = batch["fix_vectors"]
            batch_performances = batch["performances"]
            gt_qid_name.extend(batch["question_ids"])
            for fix_vectors, performances in zip(batch_fix_vectors, batch_performances):
                sample_all_metrics_rlt = []
                sample_right_answer_all_metrics_rlt = []
                sample_wrong_answer_all_metrics_rlt = []
                for index_1 in range(len(fix_vectors)):
                    fix_vector_1 = fix_vectors[index_1]
                    performance_1 = performances[index_1]
                    for index_2 in range(0, len(fix_vectors)):
                        if index_2 == index_1:
                            continue
                        fix_vector_2 = fix_vectors[index_2]
                        performance_2 = performances[index_2]
                        # calculate multimatch
                        rlt = multimatch.docomparison(fix_vector_1, fix_vector_2, screensize=[320, 240])
                        all_metrics_rlt_1vs1 = rlt
                        if np.any(np.isnan(all_metrics_rlt_1vs1)):
                            continue

                        # perform scanmatch
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array([list(_) for _ in list(fix_vector_1)])
                        np_fix_vector_2 = np.array([list(_) for _ in list(fix_vector_2)])
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                        all_metrics_rlt_1vs1.append(score_wd)
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                        all_metrics_rlt_1vs1.append(score_wod)

                        # perfrom SED
                        sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                        all_metrics_rlt_1vs1.append(sed)

                        # perfrom STDE
                        stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                        all_metrics_rlt_1vs1.append(stde)

                        sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                        # collect the right answers in the same group
                        if performance_1 == True and performance_2 == True:
                            sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)
                        # collect the wrong answers in the same group
                        elif performance_1 == False and performance_2 == False:
                            sample_wrong_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
                collect_right_answer_all_metrics_rlts.append(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
                collect_wrong_answer_all_metrics_rlts.append(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))

                if sample_right_answer_all_metrics_rlt!=[]:
                    good_scores_of_each_images.append(
                        list(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    good_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))
                if sample_wrong_answer_all_metrics_rlt != []:
                    poor_scores_of_each_images.append(
                        list(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    poor_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))

            pbar.update(1)
    collect_all_metrics_rlts = [_ for _ in collect_all_metrics_rlts if _ != []]
    collect_right_answer_all_metrics_rlts = [_ for _ in collect_right_answer_all_metrics_rlts if _ != []]
    collect_wrong_answer_all_metrics_rlts = [_ for _ in collect_wrong_answer_all_metrics_rlts if _ != []]

    all_metrics_rlts = np.concatenate(collect_all_metrics_rlts, axis=0)
    right_answer_all_metrics_rlts = np.concatenate(collect_right_answer_all_metrics_rlts, axis=0)
    wrong_answer_all_metrics_rlts = np.concatenate(collect_wrong_answer_all_metrics_rlts, axis=0)

    all_metrics = all_metrics_rlts.mean(0)
    right_answer_all_metrics = right_answer_all_metrics_rlts.mean(0)
    wrong_answer_all_metrics = wrong_answer_all_metrics_rlts.mean(0)

    all_metrics_std = all_metrics_rlts.std(0)
    right_answer_all_metrics_std = right_answer_all_metrics_rlts.std(0)
    wrong_answer_all_metrics_std = wrong_answer_all_metrics_rlts.std(0)

    summary_mean = [all_metrics, right_answer_all_metrics, wrong_answer_all_metrics]
    summary_std = [all_metrics_std, right_answer_all_metrics_std, wrong_answer_all_metrics_std]

    collected_rlts = [collect_all_metrics_rlts, collect_right_answer_all_metrics_rlts,
                      collect_wrong_answer_all_metrics_rlts]

    for index in range(len(collected_rlts)):
        specific_rlts = collected_rlts[index]
        tmp = np.concatenate([np.concatenate([[specific_rlts[index][:, 7].min(keepdims=True),
                                               specific_rlts[index][:, 8].max(keepdims=True)]]).transpose((1, 0))
                              for index in range(len(specific_rlts))], axis=0)
        summary_mean[index] = np.concatenate([summary_mean[index], tmp.mean(0)], axis=0)
        summary_std[index] = np.concatenate([summary_std[index], tmp.std(0)], axis=0)

    human_metrics = dict()
    human_metrics_std = dict()
    categories = ["all", "right_answer", "wrong_answer"]
    for category, specific_mean in zip(categories, summary_mean):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_mean[0]
        multimatch_human_metrics["direction"] = specific_mean[1]
        multimatch_human_metrics["length"] = specific_mean[2]
        multimatch_human_metrics["position"] = specific_mean[3]
        multimatch_human_metrics["duration"] = specific_mean[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_mean[5]
        scanmatch_human_metrics["with duration"] = specific_mean[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_mean[7]
        VAME_human_metrics["STDE"] = specific_mean[8]
        VAME_human_metrics["SED_best"] = specific_mean[9]
        VAME_human_metrics["STDE_best"] = specific_mean[10]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics[category] = specific_metrics

    for category, specific_std in zip(categories, summary_std):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_std[0]
        multimatch_human_metrics["direction"] = specific_std[1]
        multimatch_human_metrics["length"] = specific_std[2]
        multimatch_human_metrics["position"] = specific_std[3]
        multimatch_human_metrics["duration"] = specific_std[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_std[5]
        scanmatch_human_metrics["with duration"] = specific_std[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_std[7]
        VAME_human_metrics["STDE"] = specific_std[8]
        VAME_human_metrics["SED_best"] = specific_std[9]
        VAME_human_metrics["STDE_best"] = specific_std[10]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics_std[category] = specific_metrics

    scores_of_each_images_dict = dict()
    for name, good_score, poor_score in zip(gt_qid_name, good_scores_of_each_images, poor_scores_of_each_images):
        scores_of_each_images_dict[name] = dict([(True, good_score), (False, poor_score)])
    return human_metrics, human_metrics_std, scores_of_each_images_dict



def human_evaluation_inter(dataloader):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []
    good_scores_of_each_images = []
    poor_scores_of_each_images = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    gt_qid_name = list()
    with tqdm(total=len(dataloader)) as pbar:
        for i_batch, batch in enumerate(dataloader):
            batch_fix_vectors = batch["fix_vectors"]
            batch_performances = batch["performances"]
            gt_qid_name.extend(batch["question_ids"])
            for fix_vectors, performances in zip(batch_fix_vectors, batch_performances):
                sample_all_metrics_rlt = []
                sample_right_answer_all_metrics_rlt = []
                sample_wrong_answer_all_metrics_rlt = []
                for index_1 in range(len(fix_vectors)):
                    fix_vector_1 = fix_vectors[index_1]
                    performance_1 = performances[index_1]
                    for index_2 in range(0, len(fix_vectors)):
                        # if index_2 == index_1:
                        #     continue
                        fix_vector_2 = fix_vectors[index_2]
                        performance_2 = performances[index_2]
                        if performance_1 == performance_2:
                            continue
                        # calculate multimatch
                        rlt = multimatch.docomparison(fix_vector_1, fix_vector_2, screensize=[320, 240])
                        all_metrics_rlt_1vs1 = rlt
                        if np.any(np.isnan(all_metrics_rlt_1vs1)):
                            continue

                        # perform scanmatch
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array([list(_) for _ in list(fix_vector_1)])
                        np_fix_vector_2 = np.array([list(_) for _ in list(fix_vector_2)])
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                        all_metrics_rlt_1vs1.append(score_wd)
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                        all_metrics_rlt_1vs1.append(score_wod)

                        # perfrom SED
                        sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                        all_metrics_rlt_1vs1.append(sed)

                        # perfrom STDE
                        stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                        all_metrics_rlt_1vs1.append(stde)

                        sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                        # collect the right/wrong answers in the same group
                        sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)


                collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
                collect_right_answer_all_metrics_rlts.append(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
                collect_wrong_answer_all_metrics_rlts.append(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))

                if sample_right_answer_all_metrics_rlt!=[]:
                    good_scores_of_each_images.append(
                        list(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    good_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))
                if sample_wrong_answer_all_metrics_rlt != []:
                    poor_scores_of_each_images.append(
                        list(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    poor_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))

            pbar.update(1)


    scores_of_each_images_dict = dict()
    for name, good_score in zip(gt_qid_name, good_scores_of_each_images):
        scores_of_each_images_dict[name] = dict([("good_to_poor", good_score)])
    return scores_of_each_images_dict


def human_evaluation_upperbound(dataloader):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []
    good_scores_of_each_images = []
    poor_scores_of_each_images = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    gt_image_name = list()
    with tqdm(total=len(dataloader)) as pbar:
        for i_batch, batch in enumerate(dataloader):
            batch_fix_vectors = batch["fix_vectors"]
            batch_performances = batch["performances"]
            gt_image_name.extend(batch["img_names"])
            for fix_vectors, performances in zip(batch_fix_vectors, batch_performances):
                sample_all_metrics_rlt = []
                sample_right_answer_all_metrics_rlt = []
                sample_wrong_answer_all_metrics_rlt = []
                for index_1 in range(len(fix_vectors)):
                    fix_vector_1 = fix_vectors[index_1]
                    performance_1 = performances[index_1]
                    for index_2 in range(0, len(fix_vectors)):
                        if index_2 == index_1:
                            continue
                        fix_vector_2 = fix_vectors[index_2]
                        performance_2 = performances[index_2]
                        # calculate multimatch
                        rlt = multimatch.docomparison(fix_vector_1, fix_vector_2, screensize=[320, 240])
                        all_metrics_rlt_1vs1 = rlt
                        if np.any(np.isnan(all_metrics_rlt_1vs1)):
                            continue

                        # perform scanmatch
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array([list(_) for _ in list(fix_vector_1)])
                        np_fix_vector_2 = np.array([list(_) for _ in list(fix_vector_2)])
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                        all_metrics_rlt_1vs1.append(score_wd)
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                        all_metrics_rlt_1vs1.append(score_wod)

                        # perfrom SED
                        sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                        all_metrics_rlt_1vs1.append(sed)

                        # perfrom STDE
                        stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                        all_metrics_rlt_1vs1.append(stde)

                        sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                        # collect the right answers in the same group
                        if performance_1 == True and performance_2 == True:
                            sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)
                        # collect the wrong answers in the same group
                        elif performance_1 == False and performance_2 == False:
                            sample_wrong_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
                collect_right_answer_all_metrics_rlts.append(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
                collect_wrong_answer_all_metrics_rlts.append(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))

                if sample_right_answer_all_metrics_rlt!=[]:
                    good_scores_of_each_images.append(
                        list(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    good_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))
                if sample_wrong_answer_all_metrics_rlt != []:
                    poor_scores_of_each_images.append(
                        list(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float64).mean(axis=0)))
                else:
                    poor_scores_of_each_images.append(list(np.zeros((9,), dtype=np.float64)))

            pbar.update(1)
    collect_all_metrics_rlts = [_ for _ in collect_all_metrics_rlts if _ != []]
    collect_right_answer_all_metrics_rlts = [_ for _ in collect_right_answer_all_metrics_rlts if _ != []]
    collect_wrong_answer_all_metrics_rlts = [_ for _ in collect_wrong_answer_all_metrics_rlts if _ != []]

    collected_rlts = [collect_all_metrics_rlts, collect_right_answer_all_metrics_rlts,
                      collect_wrong_answer_all_metrics_rlts]
    summary_mean = []
    summary_std = []
    for index in range(len(collected_rlts)):
        specific_rlts = collected_rlts[index]
        temp = []
        for ii in range(len(specific_rlts)):
            tmp = specific_rlts[ii]
            tmp[:, 7] *= -1
            tmp = tmp.max(axis=0)
            tmp[7] *= -1
            temp.append(tmp)
        summary_mean.append(np.array(temp).mean(0))
        summary_std.append(np.array(temp).std(0))

    human_metrics = dict()
    human_metrics_std = dict()
    categories = ["all", "right_answer", "wrong_answer"]
    for category, specific_mean in zip(categories, summary_mean):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_mean[0]
        multimatch_human_metrics["direction"] = specific_mean[1]
        multimatch_human_metrics["length"] = specific_mean[2]
        multimatch_human_metrics["position"] = specific_mean[3]
        multimatch_human_metrics["duration"] = specific_mean[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_mean[5]
        scanmatch_human_metrics["with duration"] = specific_mean[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_mean[7]
        VAME_human_metrics["STDE"] = specific_mean[8]
        VAME_human_metrics["SED_best"] = specific_mean[7]
        VAME_human_metrics["STDE_best"] = specific_mean[8]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics[category] = specific_metrics

    for category, specific_std in zip(categories, summary_std):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_std[0]
        multimatch_human_metrics["direction"] = specific_std[1]
        multimatch_human_metrics["length"] = specific_std[2]
        multimatch_human_metrics["position"] = specific_std[3]
        multimatch_human_metrics["duration"] = specific_std[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_std[5]
        scanmatch_human_metrics["with duration"] = specific_std[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_std[7]
        VAME_human_metrics["STDE"] = specific_std[8]
        VAME_human_metrics["SED_best"] = specific_std[7]
        VAME_human_metrics["STDE_best"] = specific_std[8]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics_std[category] = specific_metrics

    scores_of_each_images_dict = dict()
    for name, good_score, poor_score in zip(gt_image_name, good_scores_of_each_images, poor_scores_of_each_images):
        scores_of_each_images_dict[name] = dict([(True, good_score), (False, poor_score)])
    return human_metrics, human_metrics_std, scores_of_each_images_dict


def human_evaluation_mismatch(dataloader):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)

    with tqdm(total=len(dataloader)) as pbar:
        for i_batch, batch in enumerate(dataloader):
            batch_fix_vectors = batch["fix_vectors"]
            batch_performances = batch["performances"]
            for fix_vectors, performances in zip(batch_fix_vectors, batch_performances):
                sample_all_metrics_rlt = []
                sample_right_answer_all_metrics_rlt = []
                sample_wrong_answer_all_metrics_rlt = []
                for index_1 in range(len(fix_vectors)):
                    fix_vector_1 = fix_vectors[index_1]
                    performance_1 = performances[index_1]
                    for index_2 in range(0, len(fix_vectors)):
                        if index_2 == index_1:
                            continue
                        fix_vector_2 = fix_vectors[index_2]
                        performance_2 = performances[index_2]
                        # calculate multimatch
                        rlt = multimatch.docomparison(fix_vector_1, fix_vector_2, screensize=[320, 240])
                        all_metrics_rlt_1vs1 = rlt
                        if np.any(np.isnan(all_metrics_rlt_1vs1)):
                            continue

                        # perform scanmatch
                        # we need to transform the scale of time from s to ms
                        # with duration
                        np_fix_vector_1 = np.array([list(_) for _ in list(fix_vector_1)])
                        np_fix_vector_2 = np.array([list(_) for _ in list(fix_vector_2)])
                        np_fix_vector_1[:, -1] *= 1000
                        np_fix_vector_2[:, -1] *= 1000
                        sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                        all_metrics_rlt_1vs1.append(score_wd)
                        # without duration
                        sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                        sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                        (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                        all_metrics_rlt_1vs1.append(score_wod)

                        # perfrom SED
                        sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                        all_metrics_rlt_1vs1.append(sed)

                        # perfrom STDE
                        stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                        all_metrics_rlt_1vs1.append(stde)

                        sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                        # collect the right answers in the same group
                        if performance_1 == True and performance_2 == False:
                            sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)
                        # collect the wrong answers in the same group
                        elif performance_1 == False and performance_2 == True:
                            sample_wrong_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
                collect_right_answer_all_metrics_rlts.append(np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
                collect_wrong_answer_all_metrics_rlts.append(np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))

            pbar.update(1)
    collect_all_metrics_rlts = [_ for _ in collect_all_metrics_rlts if _ != []]
    collect_right_answer_all_metrics_rlts = [_ for _ in collect_right_answer_all_metrics_rlts if _ != []]
    collect_wrong_answer_all_metrics_rlts = [_ for _ in collect_wrong_answer_all_metrics_rlts if _ != []]

    all_metrics_rlts = np.concatenate(collect_all_metrics_rlts, axis=0)
    right_answer_all_metrics_rlts = np.concatenate(collect_right_answer_all_metrics_rlts, axis=0)
    wrong_answer_all_metrics_rlts = np.concatenate(collect_wrong_answer_all_metrics_rlts, axis=0)

    all_metrics = all_metrics_rlts.mean(0)
    right_answer_all_metrics = right_answer_all_metrics_rlts.mean(0)
    wrong_answer_all_metrics = wrong_answer_all_metrics_rlts.mean(0)

    all_metrics_std = all_metrics_rlts.std(0)
    right_answer_all_metrics_std = right_answer_all_metrics_rlts.std(0)
    wrong_answer_all_metrics_std = wrong_answer_all_metrics_rlts.std(0)

    summary_mean = [all_metrics, right_answer_all_metrics, wrong_answer_all_metrics]
    summary_std = [all_metrics_std, right_answer_all_metrics_std, wrong_answer_all_metrics_std]

    collected_rlts = [collect_all_metrics_rlts, collect_right_answer_all_metrics_rlts,
                      collect_wrong_answer_all_metrics_rlts]

    for index in range(len(collected_rlts)):
        specific_rlts = collected_rlts[index]
        tmp = np.concatenate([np.concatenate([[specific_rlts[index][:, 7].min(keepdims=True),
                                               specific_rlts[index][:, 8].max(keepdims=True)]]).transpose((1, 0))
                              for index in range(len(specific_rlts))], axis=0)
        summary_mean[index] = np.concatenate([summary_mean[index], tmp.mean(0)], axis=0)
        summary_std[index] = np.concatenate([summary_std[index], tmp.std(0)], axis=0)

    human_metrics = dict()
    human_metrics_std = dict()
    categories = ["all", "right_answer", "wrong_answer"]
    for category, specific_mean in zip(categories, summary_mean):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_mean[0]
        multimatch_human_metrics["direction"] = specific_mean[1]
        multimatch_human_metrics["length"] = specific_mean[2]
        multimatch_human_metrics["position"] = specific_mean[3]
        multimatch_human_metrics["duration"] = specific_mean[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_mean[5]
        scanmatch_human_metrics["with duration"] = specific_mean[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_mean[7]
        VAME_human_metrics["STDE"] = specific_mean[8]
        VAME_human_metrics["SED_best"] = specific_mean[9]
        VAME_human_metrics["STDE_best"] = specific_mean[10]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics[category] = specific_metrics

    for category, specific_std in zip(categories, summary_std):
        specific_metrics = dict()

        multimatch_human_metrics = dict()
        multimatch_human_metrics["vector"] = specific_std[0]
        multimatch_human_metrics["direction"] = specific_std[1]
        multimatch_human_metrics["length"] = specific_std[2]
        multimatch_human_metrics["position"] = specific_std[3]
        multimatch_human_metrics["duration"] = specific_std[4]
        specific_metrics["MultiMatch"] = multimatch_human_metrics

        scanmatch_human_metrics = dict()
        scanmatch_human_metrics["w/o duration"] = specific_std[5]
        scanmatch_human_metrics["with duration"] = specific_std[6]
        specific_metrics["ScanMatch"] = scanmatch_human_metrics

        VAME_human_metrics = dict()
        VAME_human_metrics["SED"] = specific_std[7]
        VAME_human_metrics["STDE"] = specific_std[8]
        VAME_human_metrics["SED_best"] = specific_std[9]
        VAME_human_metrics["STDE_best"] = specific_std[10]
        specific_metrics["VAME"] = VAME_human_metrics

        human_metrics_std[category] = specific_metrics

    return human_metrics, human_metrics_std


def evaluation(gt_fix_vectors, predict_fix_vectors, all_performances, is_eliminating_nan=True):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)

    scores_of_each_images = []
    with tqdm(total=len(gt_fix_vectors)) as pbar:
        for index in range(len(gt_fix_vectors)):
            gt_fix_vector = gt_fix_vectors[index]
            predict_fix_vector = predict_fix_vectors[index]
            performances = all_performances[index]
            sample_all_metrics_rlt = []
            sample_right_answer_all_metrics_rlt = []
            sample_wrong_answer_all_metrics_rlt = []
            for inner_index in range(len(gt_fix_vector)):
                inner_gt_fix_vector = gt_fix_vector[inner_index]
                # calculate multimatch
                rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])
                all_metrics_rlt_1vs1 = rlt
                if np.any(np.isnan(all_metrics_rlt_1vs1)):
                    continue

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
                all_metrics_rlt_1vs1.append(score_wd)
                # without duration
                sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                all_metrics_rlt_1vs1.append(score_wod)

                # perfrom SED
                sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                all_metrics_rlt_1vs1.append(sed)

                # perfrom STDE
                stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                all_metrics_rlt_1vs1.append(stde)

                sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                # collect the right answers in the same group
                if performances[inner_index] == True:
                    sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)
                # collect the wrong answers in the same group
                else:
                    sample_wrong_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)

            collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
            collect_right_answer_all_metrics_rlts.append(
                np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
            collect_wrong_answer_all_metrics_rlts.append(
                np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))
            pbar.update(1)

    collect_all_metrics_rlts = [_ for _ in collect_all_metrics_rlts if _ != []]
    collect_right_answer_all_metrics_rlts = [_ for _ in collect_right_answer_all_metrics_rlts if _ != []]
    collect_wrong_answer_all_metrics_rlts = [_ for _ in collect_wrong_answer_all_metrics_rlts if _ != []]

    all_metrics_rlts = np.concatenate(collect_all_metrics_rlts, axis=0)
    right_answer_all_metrics_rlts = np.concatenate(collect_right_answer_all_metrics_rlts, axis=0)
    wrong_answer_all_metrics_rlts = np.concatenate(collect_wrong_answer_all_metrics_rlts, axis=0)

    all_metrics = all_metrics_rlts.mean(0)
    right_answer_all_metrics = right_answer_all_metrics_rlts.mean(0)
    wrong_answer_all_metrics = wrong_answer_all_metrics_rlts.mean(0)

    all_metrics_std = all_metrics_rlts.std(0)
    right_answer_all_metrics_std = right_answer_all_metrics_rlts.std(0)
    wrong_answer_all_metrics_std = wrong_answer_all_metrics_rlts.std(0)

    summary_mean = [all_metrics, right_answer_all_metrics, wrong_answer_all_metrics]
    summary_std = [all_metrics_std, right_answer_all_metrics_std, wrong_answer_all_metrics_std]

    collected_rlts = [collect_all_metrics_rlts, collect_right_answer_all_metrics_rlts,
                      collect_wrong_answer_all_metrics_rlts]

    for index in range(len(collected_rlts)):
        specific_rlts = collected_rlts[index]
        tmp = np.concatenate([np.concatenate([[specific_rlts[index][:, 7].min(keepdims=True),
                                               specific_rlts[index][:, 8].max(keepdims=True)]]).transpose((1, 0))
                              for index in range(len(specific_rlts))], axis=0)
        summary_mean[index] = np.concatenate([summary_mean[index], tmp.mean(0)], axis=0)
        summary_std[index] = np.concatenate([summary_std[index], tmp.std(0)], axis=0)

    cur_metrics = dict()
    cur_metrics_std = dict()
    categories = ["all", "right_answer", "wrong_answer"]
    for category, specific_mean in zip(categories, summary_mean):
        specific_metrics = dict()

        multimatch_cur_metrics = dict()
        multimatch_cur_metrics["vector"] = specific_mean[0]
        multimatch_cur_metrics["direction"] = specific_mean[1]
        multimatch_cur_metrics["length"] = specific_mean[2]
        multimatch_cur_metrics["position"] = specific_mean[3]
        multimatch_cur_metrics["duration"] = specific_mean[4]
        specific_metrics["MultiMatch"] = multimatch_cur_metrics

        scanmatch_cur_metrics = dict()
        scanmatch_cur_metrics["w/o duration"] = specific_mean[5]
        scanmatch_cur_metrics["with duration"] = specific_mean[6]
        specific_metrics["ScanMatch"] = scanmatch_cur_metrics

        VAME_cur_metrics = dict()
        VAME_cur_metrics["SED"] = specific_mean[7]
        VAME_cur_metrics["STDE"] = specific_mean[8]
        VAME_cur_metrics["SED_best"] = specific_mean[9]
        VAME_cur_metrics["STDE_best"] = specific_mean[10]
        specific_metrics["VAME"] = VAME_cur_metrics

        cur_metrics[category] = specific_metrics

    for category, specific_std in zip(categories, summary_std):
        specific_metrics = dict()

        multimatch_cur_metrics = dict()
        multimatch_cur_metrics["vector"] = specific_std[0]
        multimatch_cur_metrics["direction"] = specific_std[1]
        multimatch_cur_metrics["length"] = specific_std[2]
        multimatch_cur_metrics["position"] = specific_std[3]
        multimatch_cur_metrics["duration"] = specific_std[4]
        specific_metrics["MultiMatch"] = multimatch_cur_metrics

        scanmatch_cur_metrics = dict()
        scanmatch_cur_metrics["w/o duration"] = specific_std[5]
        scanmatch_cur_metrics["with duration"] = specific_std[6]
        specific_metrics["ScanMatch"] = scanmatch_cur_metrics

        VAME_cur_metrics = dict()
        VAME_cur_metrics["SED"] = specific_std[7]
        VAME_cur_metrics["STDE"] = specific_std[8]
        VAME_cur_metrics["SED_best"] = specific_std[9]
        VAME_cur_metrics["STDE_best"] = specific_std[10]
        specific_metrics["VAME"] = VAME_cur_metrics

        cur_metrics_std[category] = specific_metrics

    return cur_metrics, cur_metrics_std


def evaluation_performance_related(gt_fix_vectors, predict_fix_vectors, all_performances, all_allocated_performances):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE

    collect_all_metrics_rlts = []
    collect_right_answer_all_metrics_rlts = []
    collect_wrong_answer_all_metrics_rlts = []
    # allocated_performances = all_allocated_performances[0]
    allocated_performances = all_allocated_performances

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((240, 320, 3), dtype=np.float32)

    scores_of_each_images = []
    with tqdm(total=len(gt_fix_vectors)) as pbar:
        for index in range(len(gt_fix_vectors)):
            gt_fix_vector = gt_fix_vectors[index]
            predict_fix_vector = predict_fix_vectors[index]
            performances = all_performances[index]
            sample_all_metrics_rlt = []
            sample_right_answer_all_metrics_rlt = []
            sample_wrong_answer_all_metrics_rlt = []
            # scores_of_given_image = []
            for inner_index in range(len(gt_fix_vector)):
                inner_gt_fix_vector = gt_fix_vector[inner_index]
                # calculate multimatch
                rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])
                all_metrics_rlt_1vs1 = rlt
                if np.any(np.isnan(all_metrics_rlt_1vs1)):
                    continue

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
                all_metrics_rlt_1vs1.append(score_wd)
                # without duration
                sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                all_metrics_rlt_1vs1.append(score_wod)

                # perfrom SED
                sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                all_metrics_rlt_1vs1.append(sed)

                # perfrom STDE
                stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                all_metrics_rlt_1vs1.append(stde)

                sample_all_metrics_rlt.append(all_metrics_rlt_1vs1)

                # collect the right answers in the same group
                # if performances[inner_index] == True and allocated_performances[index % len(allocated_performances)] == True:
                if performances[inner_index] == True and allocated_performances[index] == True:
                    sample_right_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)
                # collect the wrong answers in the same group
                # elif performances[inner_index] == False and allocated_performances[index % len(allocated_performances)] == False:
                elif performances[inner_index] == False and allocated_performances[index] == False:
                    sample_wrong_answer_all_metrics_rlt.append(all_metrics_rlt_1vs1)

            collect_all_metrics_rlts.append(np.array(sample_all_metrics_rlt, dtype=np.float32))
            collect_right_answer_all_metrics_rlts.append(
                np.array(sample_right_answer_all_metrics_rlt, dtype=np.float32))
            collect_wrong_answer_all_metrics_rlts.append(
                np.array(sample_wrong_answer_all_metrics_rlt, dtype=np.float32))
            if allocated_performances[index] == True:
                if sample_right_answer_all_metrics_rlt != []:
                    current_score = list(np.array(sample_right_answer_all_metrics_rlt).mean(axis=0))
                else:
                    current_score = list(np.zeros((9,), dtype=np.float64))
                scores_of_each_images.append(current_score)
            else:
                if sample_wrong_answer_all_metrics_rlt != []:
                    current_score = list(np.array(sample_wrong_answer_all_metrics_rlt).mean(axis=0))
                else:
                    current_score = list(np.zeros((9,), dtype=np.float64))
                scores_of_each_images.append(current_score)
            pbar.update(1)

    collect_all_metrics_rlts = [_ for _ in collect_all_metrics_rlts if _ != []]
    collect_right_answer_all_metrics_rlts = [_ for _ in collect_right_answer_all_metrics_rlts if _ != []]
    collect_wrong_answer_all_metrics_rlts = [_ for _ in collect_wrong_answer_all_metrics_rlts if _ != []]

    all_metrics_rlts = np.concatenate(collect_all_metrics_rlts, axis=0)
    right_answer_all_metrics_rlts = np.concatenate(collect_right_answer_all_metrics_rlts, axis=0)
    wrong_answer_all_metrics_rlts = np.concatenate(collect_wrong_answer_all_metrics_rlts, axis=0)

    all_metrics = all_metrics_rlts.mean(0)
    right_answer_all_metrics = right_answer_all_metrics_rlts.mean(0)
    wrong_answer_all_metrics = wrong_answer_all_metrics_rlts.mean(0)

    all_metrics_std = all_metrics_rlts.std(0)
    right_answer_all_metrics_std = right_answer_all_metrics_rlts.std(0)
    wrong_answer_all_metrics_std = wrong_answer_all_metrics_rlts.std(0)

    summary_mean = [all_metrics, right_answer_all_metrics, wrong_answer_all_metrics]
    summary_std = [all_metrics_std, right_answer_all_metrics_std, wrong_answer_all_metrics_std]

    collected_rlts = [collect_all_metrics_rlts, collect_right_answer_all_metrics_rlts,
                      collect_wrong_answer_all_metrics_rlts]

    for index in range(len(collected_rlts)):
        specific_rlts = collected_rlts[index]
        tmp = np.concatenate([np.concatenate([[specific_rlts[index][:, 7].min(keepdims=True),
                                               specific_rlts[index][:, 8].max(keepdims=True)]]).transpose((1, 0))
                              for index in range(len(specific_rlts))], axis=0)
        summary_mean[index] = np.concatenate([summary_mean[index], tmp.mean(0)], axis=0)
        summary_std[index] = np.concatenate([summary_std[index], tmp.std(0)], axis=0)

    cur_metrics = dict()
    cur_metrics_std = dict()
    categories = ["all", "right_answer", "wrong_answer"]
    for category, specific_mean in zip(categories, summary_mean):
        specific_metrics = dict()

        multimatch_cur_metrics = dict()
        multimatch_cur_metrics["vector"] = specific_mean[0]
        multimatch_cur_metrics["direction"] = specific_mean[1]
        multimatch_cur_metrics["length"] = specific_mean[2]
        multimatch_cur_metrics["position"] = specific_mean[3]
        multimatch_cur_metrics["duration"] = specific_mean[4]
        specific_metrics["MultiMatch"] = multimatch_cur_metrics

        scanmatch_cur_metrics = dict()
        scanmatch_cur_metrics["w/o duration"] = specific_mean[5]
        scanmatch_cur_metrics["with duration"] = specific_mean[6]
        specific_metrics["ScanMatch"] = scanmatch_cur_metrics

        VAME_cur_metrics = dict()
        VAME_cur_metrics["SED"] = specific_mean[7]
        VAME_cur_metrics["STDE"] = specific_mean[8]
        VAME_cur_metrics["SED_best"] = specific_mean[9]
        VAME_cur_metrics["STDE_best"] = specific_mean[10]
        specific_metrics["VAME"] = VAME_cur_metrics

        cur_metrics[category] = specific_metrics

    for category, specific_std in zip(categories, summary_std):
        specific_metrics = dict()

        multimatch_cur_metrics = dict()
        multimatch_cur_metrics["vector"] = specific_std[0]
        multimatch_cur_metrics["direction"] = specific_std[1]
        multimatch_cur_metrics["length"] = specific_std[2]
        multimatch_cur_metrics["position"] = specific_std[3]
        multimatch_cur_metrics["duration"] = specific_std[4]
        specific_metrics["MultiMatch"] = multimatch_cur_metrics

        scanmatch_cur_metrics = dict()
        scanmatch_cur_metrics["w/o duration"] = specific_std[5]
        scanmatch_cur_metrics["with duration"] = specific_std[6]
        specific_metrics["ScanMatch"] = scanmatch_cur_metrics

        VAME_cur_metrics = dict()
        VAME_cur_metrics["SED"] = specific_std[7]
        VAME_cur_metrics["STDE"] = specific_std[8]
        VAME_cur_metrics["SED_best"] = specific_std[9]
        VAME_cur_metrics["STDE_best"] = specific_std[10]
        specific_metrics["VAME"] = VAME_cur_metrics

        cur_metrics_std[category] = specific_metrics

    return cur_metrics, cur_metrics_std, scores_of_each_images

def pairs_multimatch_eval(gt_fix_vectors, predict_fix_vectors, is_eliminating_nan=True):
    pairs_summary_metric = []
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_rlts = []
        for inner_index in range(len(gt_fix_vector)):
            inner_gt_fix_vector = gt_fix_vector[inner_index]
            rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])
            collect_rlts.append(rlt)
        collect_rlts = np.array(collect_rlts)
        if is_eliminating_nan:
            collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
        metric_mean = np.sum(collect_rlts, axis=0) / len(gt_fix_vector)
        pairs_summary_metric.append(metric_mean)

    return np.array(pairs_summary_metric)

def pairs_scanmatch_eval(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration):
    pairs_summary_metric = []
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
            collect_rlts.append([score_wod, score_wd])

        collect_rlts = np.array(collect_rlts)
        metric_mean = np.sum(collect_rlts, axis=0) / len(gt_fix_vector)
        pairs_summary_metric.append(metric_mean)

    return np.array(pairs_summary_metric)


def pairs_VAME_eval(gt_fix_vectors, predict_fix_vectors, is_eliminating_nan=True):
    pairs_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_SED_rlts = []
        collect_STDE_rlts = []
        for inner_index in range(len(gt_fix_vector)):
            inner_gt_fix_vector = gt_fix_vector[inner_index]

            np_fix_vector_1 = np.array([list(_) for _ in list(inner_gt_fix_vector)])
            np_fix_vector_2 = np.array([list(_) for _ in list(predict_fix_vector)])
            np_fix_vector_1[:, -1] *= 1000
            np_fix_vector_2[:, -1] *= 1000

            # perfrom SED
            sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
            collect_SED_rlts.append(sed)

            # perfrom STDE
            stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
            collect_STDE_rlts.append(stde)

        collect_SED_rlts = np.array(collect_SED_rlts)
        collect_STDE_rlts = np.array(collect_STDE_rlts)

        metric_value = np.zeros((4,), dtype=np.float32)
        metric_value[0] = collect_SED_rlts.mean()
        metric_value[1] = collect_SED_rlts.min()
        metric_value[2] = collect_STDE_rlts.mean()
        metric_value[3] = collect_STDE_rlts.max()

        pairs_summary_metric.append(metric_value)

    return np.array(pairs_summary_metric)

def pairs_eval(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration,
               is_eliminating_nan=True):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE
    pairs_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_rlts = []
        for inner_index in range(len(gt_fix_vector)):
            inner_gt_fix_vector = gt_fix_vector[inner_index]
            rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])

            if np.any(np.isnan(rlt)):
                rlt = list(rlt)
                rlt.extend([np.nan, np.nan, np.nan, np.nan])
                collect_rlts.append(rlt)
            else:
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

                # perfrom SED
                sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                # perfrom STDE
                stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)

                rlt = list(rlt)
                rlt.extend([score_wod, score_wd, sed, stde])
                collect_rlts.append(rlt)
        collect_rlts = np.array(collect_rlts)
        if is_eliminating_nan:
            collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
        if collect_rlts.shape[0] != 0:
            metric_mean = np.sum(collect_rlts, axis=0) / len(gt_fix_vector)
            metric_value = np.zeros((11,), dtype=np.float32)
            metric_value[:7] = metric_mean[:7]
            metric_value[7] = metric_mean[7]
            metric_value[8] = metric_mean[8]
            metric_value[9] = collect_rlts[:, 7].min()
            metric_value[10] = collect_rlts[:, 8].max()
        else:
            metric_value = np.array([np.nan] * 11)
        pairs_summary_metric.append(metric_value)

    return np.array(pairs_summary_metric)


def pairs_eval_performance_related(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration,
                                   performance, given_performance, is_eliminating_nan=True):
    # [0:5]: multimatch; [5]: scanmatch_with_duration; [6]: scanmatch_without_duration
    # [7]: SED; [8]: STDE; [9]: best_SED; [10]: best_STDE
    accept_flag = True
    pairs_same_summary_metric = []
    pairs_diff_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_same_rlts = []
        collect_diff_rlts = []
        for inner_index in range(len(gt_fix_vector)):
            inner_gt_fix_vector = gt_fix_vector[inner_index]
            rlt = multimatch.docomparison(inner_gt_fix_vector, predict_fix_vector, screensize=[320, 240])

            if np.any(np.isnan(rlt)):
                rlt = list(rlt)
                rlt.extend([np.nan, np.nan, np.nan, np.nan])
            else:
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

                # perfrom SED
                sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                # perfrom STDE
                stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)

                rlt = list(rlt)
                rlt.extend([score_wod, score_wd, sed, stde])

            if performance[index][inner_index] == given_performance:
                collect_same_rlts.append(rlt)
            else:
                collect_diff_rlts.append(rlt)
        collect_same_rlts = np.array(collect_same_rlts)
        collect_diff_rlts = np.array(collect_diff_rlts)
        if is_eliminating_nan:
            if collect_same_rlts.shape[0] != 0:
                collect_same_rlts = collect_same_rlts[np.isnan(collect_same_rlts.sum(axis=1)) == False]
                if collect_same_rlts.shape[0] == 0:
                    accept_flag = False
            if collect_diff_rlts.shape[0] != 0:
                collect_diff_rlts = collect_diff_rlts[np.isnan(collect_diff_rlts.sum(axis=1)) == False]
                if collect_diff_rlts.shape[0] == 0:
                    accept_flag = False
        if collect_same_rlts.shape[0] != 0:
            metric_mean = np.sum(collect_same_rlts, axis=0) / collect_same_rlts.shape[0]
            metric_value = np.zeros((11,), dtype=np.float32)
            metric_value[:7] = metric_mean[:7]
            metric_value[7] = metric_mean[7]
            metric_value[8] = metric_mean[8]
            metric_value[9] = collect_same_rlts[:, 7].min()
            metric_value[10] = collect_same_rlts[:, 8].max()
        else:
            metric_value = np.array([np.nan] * 11)
        pairs_same_summary_metric.append(metric_value)

        if collect_diff_rlts.shape[0] != 0:
            metric_mean = np.sum(collect_diff_rlts, axis=0) / collect_diff_rlts.shape[0]
            metric_value = np.zeros((11,), dtype=np.float32)
            metric_value[:7] = metric_mean[:7]
            metric_value[7] = metric_mean[7]
            metric_value[8] = metric_mean[8]
            metric_value[9] = collect_diff_rlts[:, 7].min()
            metric_value[10] = collect_diff_rlts[:, 8].max()
        else:
            metric_value = np.array([np.nan] * 11)
        pairs_diff_summary_metric.append(metric_value)

    return np.array(pairs_same_summary_metric), np.array(pairs_diff_summary_metric), accept_flag


def pairs_eval_scanmatch_performance_related(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration,
                                             ScanMatchwithoutDuration,
                                             performance, given_performance, is_eliminating_nan=True):
    # [0]: scanmatch_with_duration; [1]: scanmatch_without_duration
    accept_flag = True
    pairs_same_summary_metric = []
    pairs_diff_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_same_rlts = []
        collect_diff_rlts = []
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

            if performance[index][inner_index] == given_performance:
                collect_same_rlts.append(rlt)
            else:
                collect_diff_rlts.append(rlt)
        collect_same_rlts = np.array(collect_same_rlts)
        collect_diff_rlts = np.array(collect_diff_rlts)
        if is_eliminating_nan:
            if collect_same_rlts.shape[0] != 0:
                collect_same_rlts = collect_same_rlts[np.isnan(collect_same_rlts.sum(axis=1)) == False]
                if collect_same_rlts.shape[0] == 0:
                    accept_flag = False
            if collect_diff_rlts.shape[0] != 0:
                collect_diff_rlts = collect_diff_rlts[np.isnan(collect_diff_rlts.sum(axis=1)) == False]
                if collect_diff_rlts.shape[0] == 0:
                    accept_flag = False
        if collect_same_rlts.shape[0] != 0:
            metric_value = np.sum(collect_same_rlts, axis=0) / collect_same_rlts.shape[0]
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_same_summary_metric.append(metric_value)

        if collect_diff_rlts.shape[0] != 0:
            metric_value = np.sum(collect_diff_rlts, axis=0) / collect_diff_rlts.shape[0]
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_diff_summary_metric.append(metric_value)

    return np.array(pairs_same_summary_metric), np.array(pairs_diff_summary_metric), accept_flag


def gtpairs_eval_scanmatch_performance_related(gt_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration,
                                               performance, is_eliminating_nan=True):
    # [0]: scanmatch_with_duration; [1]: scanmatch_without_duration
    accept_flag = True
    pairs_same_summary_metric = []
    pairs_diff_summary_metric = []
    stimulus = np.zeros((240, 320, 3), dtype=np.float32)


    gt_good_fix_vectors = []
    gt_poor_fix_vectors = []
    pairs_good_summary_metric = []
    pairs_poor_summary_metric = []
    pairs_good_vs_poor_summary_metric = []

    for gt_fix_vector, performance_val in zip(gt_fix_vectors, performance):
        gt_good_fix_vector = []
        gt_poor_fix_vector = []
        for index in range(len(performance_val)):
            if performance_val[index] == True:
                gt_good_fix_vector.append(gt_fix_vector[index])
            else:
                gt_poor_fix_vector.append(gt_fix_vector[index])
        gt_good_fix_vectors.append(gt_good_fix_vector)
        gt_poor_fix_vectors.append(gt_poor_fix_vector)


    # calculate the both good performance
    for index in range(len(gt_good_fix_vectors)):
        gt_fix_vector1 = gt_good_fix_vectors[index]
        gt_fix_vector2 = gt_good_fix_vectors[index]
        collect_rlts = []
        if len(gt_fix_vector1) <= 1:
            pass
        else:
            for index1 in range(len(gt_fix_vector1)):
                for index2 in range(index1+1, len(gt_fix_vector2)):
                    given_gt_fix_vector1 = gt_fix_vector1[index1]
                    given_gt_fix_vector2 = gt_fix_vector2[index2]
                    # perform scanmatch
                    # we need to transform the scale of time from s to ms
                    # with duration
                    np_fix_vector_1 = np.array([list(_) for _ in list(given_gt_fix_vector1)])
                    np_fix_vector_2 = np.array([list(_) for _ in list(given_gt_fix_vector2)])
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
            if collect_rlts.shape[0] != 0:
                collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
                if collect_rlts.shape[0] == 0:
                    accept_flag = False

        if collect_rlts.shape[0] != 0:
            metric_value = np.sum(collect_rlts, axis=0) / collect_rlts.shape[0]
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_good_summary_metric.append(metric_value)


    # calculate the both poor performance
    for index in range(len(gt_poor_fix_vectors)):
        gt_fix_vector1 = gt_poor_fix_vectors[index]
        gt_fix_vector2 = gt_poor_fix_vectors[index]
        collect_rlts = []
        if len(gt_fix_vector1) <= 1:
            pass
        else:
            for index1 in range(len(gt_fix_vector1)):
                for index2 in range(index1+1, len(gt_fix_vector2)):
                    given_gt_fix_vector1 = gt_fix_vector1[index1]
                    given_gt_fix_vector2 = gt_fix_vector2[index2]
                    # perform scanmatch
                    # we need to transform the scale of time from s to ms
                    # with duration
                    np_fix_vector_1 = np.array([list(_) for _ in list(given_gt_fix_vector1)])
                    np_fix_vector_2 = np.array([list(_) for _ in list(given_gt_fix_vector2)])
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
            if collect_rlts.shape[0] != 0:
                collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
                if collect_rlts.shape[0] == 0:
                    accept_flag = False

        if collect_rlts.shape[0] != 0:
            metric_value = np.sum(collect_rlts, axis=0) / collect_rlts.shape[0]
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_poor_summary_metric.append(metric_value)

    # calculate the good vs poor performance
    for index in range(len(gt_poor_fix_vectors)):
        gt_fix_vector1 = gt_good_fix_vectors[index]
        gt_fix_vector2 = gt_poor_fix_vectors[index]
        collect_rlts = []
        if len(gt_fix_vector1) <= 1 or len(gt_fix_vector2) <= 1:
            pass
        else:
            for index1 in range(len(gt_fix_vector1)):
                for index2 in range(len(gt_fix_vector2)):
                    given_gt_fix_vector1 = gt_fix_vector1[index1]
                    given_gt_fix_vector2 = gt_fix_vector2[index2]
                    # perform scanmatch
                    # we need to transform the scale of time from s to ms
                    # with duration
                    np_fix_vector_1 = np.array([list(_) for _ in list(given_gt_fix_vector1)])
                    np_fix_vector_2 = np.array([list(_) for _ in list(given_gt_fix_vector2)])
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
            if collect_rlts.shape[0] != 0:
                collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
                if collect_rlts.shape[0] == 0:
                    accept_flag = False

        if collect_rlts.shape[0] != 0:
            metric_value = np.sum(collect_rlts, axis=0) / collect_rlts.shape[0]
        else:
            metric_value = np.array([np.nan] * 2)
        pairs_good_vs_poor_summary_metric.append(metric_value)

    return np.array(pairs_good_summary_metric), np.array(pairs_poor_summary_metric), \
           np.array(pairs_good_vs_poor_summary_metric)
