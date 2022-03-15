import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from DataProcessing.DB.dal import *


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


def find_intersecting_tracks_old(tracks_range):
    """
    Given a list with the information of all tracks start & end frames, return a list with groups of tracks
    that intersect, i.e. have at least one frame that appears in both of the tracks.
    @return: a list of groups of intersecting ranges.

    Groups of intersecting tracks are calculated using the following algorithm:
    1. Sort all tracks according to the start frame - O(nlogn)
    2. Define cur_end=end frame of the first range
    3. Iterate over all sorted ranges and check if range_start > cur_max
        i. If yes, these two ranges overlap, add the current range to the group
        ii. Else, the ranges don't overlap, close the previous group and define cur_end = end frame of current range.
    """
    if not tracks_range:
        raise Exception('No tracks ranges were given!')
    sorted_start = sorted(tracks_range, key=lambda d: (d['start'], d['end']))
    cur_end = sorted_start[0].get('end')
    groups = []
    cur_group = []
    for r in sorted_start:
        if r.get('start') <= cur_end:  # ranges overlap
            cur_group.append(r.get('track_num'))
        else:  # ranges don't overlap
            groups.append(cur_group)
            cur_group = [r.get('track_num')]
            cur_end = r.get('end')
    if len(cur_group) > 1:
        groups.append(cur_group)
    return groups


def find_intersecting_tracks(tracks_range):
    """
    Given a list with the information of all tracks start & end frames, return a list with groups of tracks
    that intersect, i.e. have at least one frame that appears in both of the tracks.
    @return: a list of groups of intersecting ranges.

    Groups of intersecting tracks are calculated using the following algorithm:
    1. Sort all tracks according to the start frame - O(nlogn)
    """
    range_points = [f'{r.get("start")}_s_{r.get("track_num")}' for r in tracks_range]
    range_points.extend([f'{r.get("end")}_e_{r.get("track_num")}' for r in tracks_range])
    sorted_range = sorted(range_points, key=lambda d: int(d.split('_')[0]))
    prev_s = -1  # todo: remove this
    prev_e = np.inf  # todo: remove this
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
            prev_e = int(point_parts[0])
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


ranges = get_track_ranges('20210808082440_s0_e501')
groups = find_intersecting_tracks(ranges)
print(groups)

G = create_graph(groups)
nx.draw(G, with_labels=True)
plt.savefig('../Results/Tracks-graph.png')
plt.show()


