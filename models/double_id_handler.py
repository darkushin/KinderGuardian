import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from DataProcessing.DB.dal import *

IDS_TO_COLORS = {
    1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown', 7: 'tab:pink',
    8: 'tab:gray', 9: 'tab:olive', 10: 'tab:cyan'
}


def get_track_ranges(vid_name: str):
    """
    Given a video_name create a list of dictionaries for each track.
    Output template: [{track_num1: X, start: X, end: X}, {track_num2: Y, start: Y, end: Y}]
    """
    tracks = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}), group=Crop.track_id)]
    session = create_session()
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


def get_ids_order_for_tracks(vid_name: str = None, G: nx.Graph() = None, seed=None):
    """
    Given a video, create for every track within the video a ranked list of the possible ids according to the ReID model
    """
    # Currently, just give a random order of ids, todo: change it to be based on true ReID!
    max_degree = max([d for n, d in G.degree()])
    num_ids = G.number_of_nodes()

    # Generate a random ID ordering for every track:
    if seed:
        random.seed(seed)
    ids_rank = {node: random.sample(range(1, max_degree+1), max_degree) for node in G.nodes}
    return ids_rank


def assign_color_to_node(G: nx.Graph(), ids_rank: dict, state: str=None):
    color_map = []
    colored_nodes = {}

    if state == 'rank-1':
        # For every node take the first rank in the ids_rank (with collisions)
        for node in G:
            color = IDS_TO_COLORS[ids_rank[node][0]]
            color_map.append(color)
            colored_nodes[node] = color

    elif state == 'node-order':
        # for node in sorted(G):
        for node in G:
            for rank in ids_rank[node]:
                cur_color = IDS_TO_COLORS[rank]
                valid_color = True
                for neighbor in G.neighbors(node):
                    if colored_nodes.get(neighbor, '') == cur_color:
                        valid_color = False
                        break  # color can't be assigned, move to next color
                if valid_color:
                    colored_nodes[node] = cur_color  # no neighbors were found with this color
                    color_map.append(cur_color)
                    break

    pos = nx.spring_layout(G, seed=152)  # use seed so the resulting graph will have the same order every time
    nx.draw(G, pos=pos, node_color=color_map, with_labels=True)
    plt.show()
    print(f'Colored Nodes: {colored_nodes}')


ranges = get_track_ranges('20210808082440_s0_e501')
groups = find_intersecting_tracks(ranges)
print(groups)

G = create_graph(groups)
# nx.draw(G, with_labels=True)
# plt.savefig('../Results/Tracks-graph.png')
# plt.show()

ids_rank = get_ids_order_for_tracks(G=G, seed=152)
print(ids_rank)

assign_color_to_node(G, ids_rank, state='node-order')

