import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

import pickle

from DataProcessing.DB.dal import *
from DataProcessing.dataProcessingConstants import ID_TO_NAME

IDS_TO_COLORS = {
    1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown', 7: 'tab:pink',
    8: 'tab:gray', 9: 'tab:olive', 10: 'tab:cyan'
}

# The order in which the ids should be allocated to ids:
# - rank-1: take the id with the highest score for each track (doesn't remove double ids! should be identical to the
#   combination of reid and face)
# - sorted-rank-1: sort the nodes from highest rank-1 to lowest and allocate ids first for higher ranking nodes
# - appearance-order: iterate over the nodes in the appearance order of the tracks in the video (the order in which the
#   nodes were entered to the graph)
# - max-difference: sort the nodes from the nodes for which the difference between rank-1 and rank-2 is highest
NODES_ORDER = ['rank-1', 'sorted-rank-1', 'appearance-order', 'max-difference']


def get_track_ranges(vid_name: str, db_location: str):
    """
    Given a video_name create a list of dictionaries for each track.
    Output template: [{track_num1: X, start: X, end: X}, {track_num2: Y, start: Y, end: Y}]
    """
    tracks = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}),
                                                      group=Crop.track_id,
                                                      db_path=db_location)]
    session = create_session(db_location=db_location)
    tracks_range = []
    for track in tracks:
        min_frame = session.query(func.min(Crop.frame_num)).filter(
            and_(Crop.track_id == track, Crop.vid_name == vid_name)).all()[0]
        max_frame = session.query(func.max(Crop.frame_num)).filter(
            and_(Crop.track_id == track, Crop.vid_name == vid_name)).all()[0]
        tracks_range.append({'track_num': track, 'start': min_frame[0], 'end': max_frame[0]})
    return tracks_range


def find_intersecting_tracks(tracks_range):
    """
    Given a list with the information of all tracks start & end frames, return a list with groups of tracks
    that intersect, i.e. have at least one frame that appears in both of the tracks.

    Groups of intersecting tracks are calculated using the following algorithm:
    1. For each track i with range (m, n) extract range points of the form "m_s_i" and "n_e_i" where 'm' and 'n' are the
       start and end point respectively, 's' stands for start, 'e' stands for end and 'i' is the track number.
    2. Sort all tracks according to the point coordinate (the first part in every "x_s/e_i" point) - O(nlogn)
    3. Initialize an active_ranges list and start iterate over the sorted points list:
        i. If the current point is a starting point of a range, add it to the set and continue.
        ii. If the current point is an ending point of a range, save the current set of active ranges and remove the
            current track from the list of active ranges.
    """
    range_points = [f'{r.get("start")}_s_{r.get("track_num")}' for r in tracks_range]
    range_points.extend([f'{r.get("end")}_e_{r.get("track_num")}' for r in tracks_range])
    sorted_range = sorted(range_points, key=lambda d: int(d.split('_')[0]))
    all_groups = []
    cur_group = set()
    for i, point in enumerate(sorted_range):
        point_parts = point.split('_')
        if point_parts[1] == 's':
            cur_group.add(int(point_parts[2]))
        else:
            if i+1 < len(sorted_range):  # check if the next point is also an end point that ends in the same range
                next_point_parts = sorted_range[i+1].split('_')
                if next_point_parts[1] == 'e' and int(next_point_parts[0]) == int(point_parts[0]):  # don't close the group yet
                    continue
                else:
                    pass
            all_groups.append(copy.deepcopy(cur_group))
            cur_group.remove(int(point_parts[2]))
    return all_groups


def create_graph(groups):
    """
    Given the groups of intersecting ranges, creates a graph that has an edge between every two tracks that intersect.
    Note: every group is a complete-group of size |group_size| which is a sub-graph in the final graph.
    """
    # Create a complete sub-graph for every group and merge to the final graph:
    G = nx.Graph()
    for group in groups:
        sub_g = nx.complete_graph(group)
        G = nx.compose(G, sub_g)
    return G


def draw_graph(G: nx.Graph(), color_map: list = [], title: str = None):
    pos = nx.spring_layout(G, seed=152)  # use seed so the resulting graph will have the same order every time
    plt.figure()
    if title:
        plt.title(title)
    nx.draw(G, pos=pos, with_labels=True, node_color=color_map, cmap='tab20')
    plt.show()


def sort_by_max_difference(ids_rank):
    """
    Sort the nodes in an order such that first appear tracks for which the difference between rank-1 and rank-2 is the largest
    """
    rank_diffs = {key: track_ranks[0][1] - track_ranks[1][1] for key, track_ranks in ids_rank.items()}
    ordered_nodes = [k for k, _ in sorted(rank_diffs.items(), key=lambda item: item[1], reverse=True)]
    return ordered_nodes


def assign_color_to_node(G: nx.Graph(), ids_rank: dict, nodes_order: str = 'rank-1'):
    """
    Given a graph and the ids_rank for every node in the graph, assign colors to all nodes so that two adjacent nodes
    don't share a common color.
    Arguments:
        - G: the graph that should be colored.
        - ids_rank: a dictionary where the keys are all nodes in the graph and the value of every key is an ordered
                    list with the probabilities of every id for this node (track) in descending order.
    """
    color_map = []
    colored_nodes = {}

    # draw the initial graph before coloring
    # draw_graph(G, title='Graph without coloring')

    if nodes_order == 'rank-1':
        # Show the graph of rank-1 predictions: for every node take the first rank in the ids_rank (with collisions)
        for node in G:
            color = ids_rank[node][0][0]
            color_map.append(color)
            colored_nodes[node] = color
        # draw_graph(G, color_map, title='Rank-1 coloring')
        color_names = {k: ID_TO_NAME[v] for k, v in colored_nodes.items()}
        # print(f'Rank-1 colors: {color_names}')
        return colored_nodes

    if nodes_order == 'sorted-rank-1':
        # Sort the nodes according to rank-1 in descending order:
        ordered_nodes = [k for k, _ in sorted(ids_rank.items(), key=lambda item: item[1][0][1], reverse=True)]
    elif nodes_order == 'appearance-order':
        ordered_nodes = G.nodes
    elif nodes_order == 'max-difference':
        ordered_nodes = sort_by_max_difference(ids_rank)
    else:
        raise Exception(f'Double Booking: nodes order method not selected or invalid. Options are: {NODES_ORDER}')

    # Assign ids to tracks according to the nodes order:
    color_map = []
    colored_nodes = {}
    for node in ordered_nodes:
        for id, prob in ids_rank[node]:
            cur_color = id
            valid_color = True
            for neighbor in G.neighbors(node):
                if colored_nodes.get(neighbor, '') == cur_color:
                    valid_color = False
                    break  # color can't be assigned, move to next color
            if valid_color:
                colored_nodes[node] = cur_color  # no neighbors were found with this color
                break

    # update color map:
    for node in G:
        color_map.append(colored_nodes.get(node))

    # draw the colored graph:
    # draw_graph(G, color_map=color_map, title=nodes_order)

    color_names = {k: ID_TO_NAME[v] for k, v in colored_nodes.items()}
    # print(f'{nodes_order}: {color_names}')
    return colored_nodes


def sort_track_scores(G, tracks_scores):
    """
    Given the probability of each id for every track and the graph of this video, create a dictionary where every key
    is the track number and the value is a tuple of the form (id, id_probability) from the must likely id to the
    least likely.
    """
    sorted_track_scores = {}
    for track in G.nodes:
        probs = tracks_scores.get(track)
        if probs:
            sorted_track_scores[track] = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_track_scores


def remove_double_ids(vid_name: str, tracks_scores: dict, db_location: str, nodes_order: str):
    """
    Given a video name and the location of the DB that contains this video, create a mapping from tracks to ids, such
    that intersecting tracks can't get the same id.
    """
    # Find the ranges of all tracks in the video:
    ranges = get_track_ranges(vid_name, db_location)

    # Find intersecting tracks:
    groups = find_intersecting_tracks(ranges)

    # Generate a graph with all intersections as neighbors:
    G = create_graph(groups)

    ids_rank = sort_track_scores(G, tracks_scores)

    tracks_to_ids = assign_color_to_node(G, ids_rank, nodes_order=nodes_order)
    return tracks_to_ids


if __name__ == '__main__':
    db_location = '/home/bar_cohen/raid/dani-inference_db8.db'
    all_tracks_final_scores = pickle.load(open('/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/all-tracks-20210808111437_s45000_e45501.pkl','rb'))

    # new_id_dict = remove_double_ids('20210808111437_s45000_e45501', all_tracks_final_scores, db_location, 'max-difference')
    ids_rank = sort_track_scores(None, all_tracks_final_scores)
    new_id_dict = sort_by_max_difference(ids_rank)
    print(new_id_dict)



