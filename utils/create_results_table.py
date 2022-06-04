import pandas as pd

AVERAGE_KEYS = ['pure_reid_model', 'reid_with_maj_vote', 'face_clf_only', 'face_clf_only_tracks_with_face',
                'reid_with_face_clf_maj_vote', 'rank-1', 'sorted-rank-1', 'appearance-order', 'max-difference']


def compute_field_average(field_values, num_crops):
    weighted_average = 0
    total_crops = sum(num_crops)
    for i in range(len(field_values)):
        weighted_average += field_values[i] * num_crops[i]
    return weighted_average / total_crops


def compute_weighted_average(results_file_path):
    """
    results_file_path: the path to the file that holds the desired results. The file should have the same titles as our
    results file.
    """
    results = pd.read_csv(results_file_path)
    averages_dict = {}
    for column in AVERAGE_KEYS:
        averages_dict[column] = compute_field_average(results[column].values, results['total_crops'].values)

    results.append(averages_dict, ignore_index=True).to_csv(results_file_path)


def create_overleaf_table(results_file_path, output_location):
    results = pd.read_csv(results_file_path)

    # create the table:
    with open(output_location, 'w') as tf:
        tf.write(pd.DataFrame.to_latex(results))


if __name__ == '__main__':
    # compute_weighted_average('/Users/darkushin/Downloads/ctl-results.csv')
    create_overleaf_table('/Users/darkushin/Downloads/baseline-results.csv',
                          '/Users/darkushin/Downloads/baseline-results.tex')
