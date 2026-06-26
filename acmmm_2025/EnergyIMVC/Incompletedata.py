import numpy as np
from scipy.io import loadmat, savemat
import os

def make_data_incomplete_and_track_missing(X, missing_ratio):
    """
    Process data for each view according to the specified missing ratio, ensuring each instance appears in at least one view.
    At the same time, record and return a matrix indicating which instances are missing in which views.

    :param X: list of arrays or dict of arrays, each element is a 2D array or a dictionary of 2D arrays.
    :param missing_ratio: specified missing ratio, ranging from 0 to 1.
    :return: processed X and a boolean matrix indicating the missing positions.
    """
    if isinstance(X, dict):
        # X is a dictionary of arrays
        num_views = len(X)
        view_keys = list(X.keys())
        num_instances = X[view_keys[0]].shape[0]
    else:
        # X is a list of arrays
        num_views = len(X)
        num_instances = X[0].shape[0]

    missing_matrix = np.zeros((num_instances, num_views), dtype=bool)
    ratio = missing_ratio / num_views

    # Randomly select a protected view for each instance
    protected_views_for_instances = np.random.randint(num_views, size=num_instances)

    for view_index in range(num_views):
        if isinstance(X, dict):
            view_data = X[view_keys[view_index]]
        else:
            view_data = X[view_index]

        num_missing_instances = int(np.floor(ratio * num_instances))
        possible_indices = [i for i in range(num_instances) if protected_views_for_instances[i] != view_index]

        if len(possible_indices) < num_missing_instances:
            indices_to_remove = possible_indices
        else:
            indices_to_remove = np.random.choice(possible_indices, num_missing_instances, replace=False)

        # Set the selected instances as missing in the current view
        for idx in indices_to_remove:
            view_data[idx] = 0
            missing_matrix[idx, view_index] = True

        if isinstance(X, dict):
            X[view_keys[view_index]] = view_data
        else:
            X[view_index] = view_data

    return X, missing_matrix

def main():
    # Process Cell Array dataset
    # mat_data = loadmat('./datasets/handwritten.mat', squeeze_me=True)
    # X = mat_data['X']
    # # X = mat_data['data']
    # Y = mat_data['Y']
    # # Y = mat_data['truth']
    # Y = Y.reshape(-1, 1)
    # missing_ratio = 0.9
    #
    # # Process MSRCv1 data with missing handling and record missing information
    # X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)
    #
    # # Save the processed Cell Array data and missing matrix
    # original_filename = 'handwritten.mat'
    # original_filename_without_extension = os.path.splitext(original_filename)[0]
    # new_filename= f"{original_filename_without_extension}_missing_{missing_ratio}.mat"
    # mat_data['X'] = X_incomplete
    # mat_data['Y'] = Y
    # mat_data['missing_matrix'] = missing_matrix  # Save missing matrix
    # savemat(new_filename, mat_data)
    # print(f"Processed data and missing matrix have been saved as: {new_filename}")


    # Process non-Cell Array dataset
    mat_data = loadmat('./datasets/Fashion.mat', squeeze_me=True)
    X = {key: mat_data[key] for key in mat_data.keys() if key.startswith('X')}
    Y = mat_data['Y']
    Y = Y.reshape(-1, 1)
    missing_ratio = 0.9

    # Process non-Cell Array data with missing handling and record missing information
    X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)

    # Save the processed non-Cell Array data and missing matrix
    original_filename = 'Fashion.mat'
    original_filename_without_extension = os.path.splitext(original_filename)[0]
    new_filename = f"{original_filename_without_extension}_missing_{missing_ratio}.mat"
    mat_data.update(X_incomplete)
    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # Save missing matrix
    savemat(new_filename, mat_data)
    print(f"Processed data and missing matrix have been saved as: {new_filename}")

if __name__ == "__main__":
    main()