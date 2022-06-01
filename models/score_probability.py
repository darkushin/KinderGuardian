import os
import pickle
import sys

import pandas
import pandas as pd

from DataProcessing.dataProcessingConstants import ID_TO_NAME
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('fast-reid')

# from fastreid.config import get_cfg
# from fastreid.data import build_reid_test_loader
# from demo.predictor import FeatureExtractionDemo
# from models.track_and_reid_model import set_reid_cfgs
# from models.model_runner import get_args
# import tqdm
# import torch
# import numpy as np
# import torch.nn.functional as F
# import matplotlib.pyplot as plt


def create_feature_embeddings(reid_model, data_loader):
    feats = []
    pids = []
    camids = []
    print('Converting test data to feature vectors:')
    # Converts all images in the bounding_box_test and query to feature vectors
    for (feat, pid, camid) in tqdm.tqdm(reid_model.run_on_loader(data_loader), total=len(data_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    print("The size of test gallery is", len(pids))
    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    return q_feat, g_feat, q_pids, g_pids, q_camids, g_camids


def plot_results(scores_dict, title):
    """
    Given a scores_dict, create a bar plot for these scores.
    scores_dict: a dictionary where the keys are scores and the values are a dictionary of the type:
    {'correct': #correct, 'incorrect': #incorrect}
    """
    # convert all counters to probabilities and plot on graph
    sorted_scores = dict(sorted(scores_dict.items()))
    x_axis = sorted_scores.keys()
    x = list(x_axis)
    y_probs = [s['correct'] / (s['correct'] + s['incorrect']) for s in sorted_scores.values()]
    y_correct = [s['correct'] for s in sorted_scores.values()]
    y_incorrect = [s['incorrect'] for s in sorted_scores.values()]


    import  seaborn as sns

    df = pd.DataFrame({'score': x, 'accuracy':y_probs, 'correct':y_correct, 'incorrect':y_incorrect})
    sns.barplot(data=df, x='score',y='accuracy', color='orange')
    plt.title(f"{title}")
    plt.show()

    plt.clf()


    sns.barplot(data=df, x='score',y='correct', color='green')
    sns.barplot(data=df, x='score',y='incorrect', color='red')
    plt.ylabel('Correct / Incorrect Count')
    plt.title(f"{title}")
    plt.show()




    # create a graph that left y-axis is probability and right y-axis is counter of appearances. This graph should have
    # three bars for every score in the x-axis.
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # barWidth = 0.3
    # r1 = np.arange(len(y_probs))
    # r2 = [x + barWidth for x in r1]
    # r3 = [x + barWidth for x in r2]
    #
    # ax.bar(r1, y_probs, color='blue', width=barWidth, edgecolor='white', label='Probability')
    # ax2.bar(r2, y_correct, color='green', width=barWidth, edgecolor='white', label='Correct')
    # ax2.bar(r3, y_incorrect, color='red', width=barWidth, edgecolor='white', label='Incorrect')
    #
    # plt.xlabel('Scores', fontweight='bold')
    # plt.xticks([r + barWidth for r in range(len(y_probs))], x)
    # plt.title(title)
    # plt.legend(labels=['a','b'])
    # plt.show()

    print('daniel')


def fast_reid_example():
    args = get_args()

    # initialize reid model
    reid_cfg = set_reid_cfgs(args)
    reid_model = FeatureExtractionDemo(reid_cfg, parallel=False)

    # create feature embeddings for gallery
    test_loader, num_query = build_reid_test_loader(reid_cfg, dataset_name='DukeMTMC',
                                                    num_workers=0)  # will take the dataset given as argument
    # q_feat, g_feat, q_pids, g_pids, q_camids, g_camids = create_feature_embeddings(reid_model, test_loader)

    # pickle.dump((q_feat, g_feat, q_pids, g_pids, q_camids, g_camids),
    #             open(os.path.join('/mnt/raid1/home/bar_cohen/Scores_Probability/', 'reid_features.pkl'), 'wb'))

    q_feat, g_feat, q_pids, g_pids, q_camids, g_camids = pickle.load(
        open(os.path.join('/mnt/raid1/home/bar_cohen/Scores_Probability/', 'reid_features.pkl'), 'rb'))

    # compute cosine similarity between gallery and query features
    features = F.normalize(q_feat, p=2, dim=1)
    others = F.normalize(g_feat, p=2, dim=1)
    distmat = torch.mm(features, others.t())
    distmat = distmat.numpy()

    # take the id of the gallery feature with the highest score
    best_match_in_gallery = np.argmax(distmat, axis=1)
    predicted_pids = g_pids[best_match_in_gallery]
    scores = []
    for i in range(len(predicted_pids)):
        scores.append(distmat[i, best_match_in_gallery[i]])

    # log the score as the x value and add 1 to the corresponding true/false counter of this score
    scores_dict = {}
    for score, p_pid, q_pid in zip(scores, predicted_pids, q_pids):
        score = round(score, 3)
        if score not in scores_dict:
            scores_dict[score] = {'correct': 0, 'incorrect': 0}

        if p_pid == q_pid:
            scores_dict[score]['correct'] += 1
        else:
            scores_dict[score]['incorrect'] += 1

    plot_results(scores_dict, 'Title for the plot')


def scores_probability():
    """
    Runs on all the pre-recorded scores of tracks and computes the accuracy as a function of the score of a track.
    """
    root_dir = '/home/bar_cohen/raid/alpha_tuning/new_score_functions_31.5/'
    reid_scores_dict = {}
    face_scores_dict = {}
    for gallery in os.listdir(root_dir):
        for pkl in os.listdir(os.path.join(root_dir, gallery)):
            if '.pkl' not in pkl:
                continue
            tracks_scores = pickle.load(open(os.path.join(root_dir, gallery, pkl), 'rb'))
            for track_item in tracks_scores.items():
                track = track_item[1]
                assert len(np.unique(track.get('true_label'))) == 1
                true_label = track.get('true_label')[0]

                # update reid_scores:
                reid_scores = track.get('reid_scores')
                reid_score = max(reid_scores.values())
                reid_score = round(reid_score, 1)
                if reid_score not in reid_scores_dict:
                    reid_scores_dict[reid_score] = {'correct': 0, 'incorrect': 0}
                predicted_reid_label = ID_TO_NAME[max(reid_scores, key=reid_scores.get)]
                if true_label == predicted_reid_label:
                    reid_scores_dict[reid_score]['correct'] += 1
                else:
                    reid_scores_dict[reid_score]['incorrect'] += 1

                # update face_scores:
                face_scores = track.get('face_scores')
                if face_scores:
                    face_score = max(face_scores.values())
                    face_score = round(face_score, 1)
                    if face_score not in face_scores_dict:
                        face_scores_dict[face_score] = {'correct': 0, 'incorrect': 0}
                    predicted_face_label = ID_TO_NAME[max(face_scores, key=face_scores.get)]
                    if true_label == predicted_face_label:
                        face_scores_dict[face_score]['correct'] += 1
                    else:
                        face_scores_dict[face_score]['incorrect'] += 1

    plot_results(reid_scores_dict, 'Re-ID Scores-Acc Trend')
    plot_results(face_scores_dict, 'Face Scores-Acc Trend')


def alpha_tuning():
    root_dir = '/home/bar_cohen/raid/alpha_tuning/new_score_functions_31.5/'
    for gallery in os.listdir(root_dir):
        for pkl in os.listdir(os.path.join(root_dir, gallery)):
            if '.pkl' not in pkl:
                continue
            print(f'Creating plot for: {pkl}')
            all_accuracies = []
            alphas = np.arange(0, 1.05, 0.05)
            for alpha in alphas:
                tracks_scores = pickle.load(open(os.path.join(root_dir, gallery, pkl), 'rb'))
                total_imgs = 0
                correct_labels = 0
                for track_item in tracks_scores.items():
                    track = track_item[1]
                    reid_scores = track.get('reid_scores')
                    face_scores = track.get('face_scores')
                    assert len(np.unique(track.get('true_label'))) == 1
                    true_label = track.get('true_label')[0]
                    if face_scores:
                        final_scores = {pid : alpha*reid_score + (1-alpha) * face_score for pid, reid_score, face_score in zip(reid_scores.keys() , reid_scores.values(), face_scores.values())}
                    else:  # no face was detected, we take only reid prediction in this case
                        final_scores = reid_scores
                    predicted_label = ID_TO_NAME[max(final_scores, key=final_scores.get)]
                    num_crops = len(track.get('true_label'))
                    total_imgs += num_crops
                    if true_label == predicted_label:
                        correct_labels += num_crops
                all_accuracies.append(correct_labels / total_imgs)
            plt.scatter(alphas, all_accuracies)
            vid_name = pkl.split('_tracks')[0]
            best_alphas = alphas[np.where(all_accuracies == np.max(all_accuracies))]
            plt.title(f'Alpha Accuracies for video: {vid_name} \n Gallery: {gallery}')
            plt.xlabel(f'best alpha: {best_alphas}')
            plt.ylabel('Accuracy')
            fig_name = f'{vid_name}_{gallery}.png'
            plt.savefig(f'{os.path.join(root_dir, gallery, fig_name)}')
            plt.show()


if __name__ == '__main__':
    # alpha_tuning()
    scores_probability()

