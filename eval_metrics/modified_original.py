import math
import os
from pathlib import Path
import glob
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def stl_to_point_cloud(filename):
    try:
        point_cloud = o3d.io.read_triangle_mesh(filename)
        point_cloud = point_cloud.sample_points_poisson_disk(1000)
        return point_cloud
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return None


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def best_fit_transform(A, B):
    """
    Calculates the best-fit transform that maps points A onto points B.
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
    Output:
        T: (m+1)x(m+1) homogeneous transformation matrix
    """

    # Check if A and B have same dimensions
    assert A.shape == B.shape

    # Get number of dimensions
    m = A.shape[1]

    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation
    t = centroid_B.reshape(-1, 1) - np.dot(R, centroid_A.reshape(-1, 1))

    # Homogeneous transformation
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()

    return T


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)  # n_neighbors=1
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def iterative_closest_point(A, B, max_iterations=20, tolerance=0.001):  # tolerance=0.001
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source points
        B: Nxm numpy array of destination points
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        finalA: Aligned points A; Source points A after getting mapped to destination points B
        final_error: Sum of euclidean distances (errors) of the nearest neighbors
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):  # tqdm.tqdm(range(max_iterations)):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error (stop if error is less than specified tolerance)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, error, and mapped source points
    T = best_fit_transform(A, src[:m, :].T)
    final_error = prev_error

    # get final A
    rot = T[0:-1, 0:-1]
    t = T[:-1, -1]
    finalA = np.dot(rot, A.T).T + t

    return T, finalA, final_error, i


def intersection_over_GT2(GT, test, threshold=-0.6):  # -0.6
    # Convert to binary images based on a threshold
    # print(GT)
    GT_binary = (GT > threshold).astype(np.uint8)
    test_binary = (test > threshold).astype(np.uint8)
    # print(f"GT_binary:\n{GT_binary}")
    # print(f"test_binary:\n{test_binary}")
    # Compute the intersection of GT and test images
    intersection = np.logical_and(GT_binary, test_binary).astype(np.uint8)

    # Calculate the ratio of the intersection over GT's foreground
    if np.count_nonzero(GT_binary) == 0:
        return 0  # Handle division by zero if GT has no foreground pixels
    return np.count_nonzero(intersection) / np.count_nonzero(GT_binary)


def point_cloud_distance(pc1, pc2):
    """
    Calculates the average minimum distance from each point in pc1 to the closest point in pc2.

    Parameters:
    - pc1: numpy array of shape (N, 3) representing the first point cloud.
    - pc2: numpy array of shape (M, 3) representing the second point cloud.

    Returns:
    - avg_distance: The average of the minimum distances from points in pc1 to points in pc2.
    """
    # Expand dimensions for broadcasting to calculate distances between each pair of points
    pc1_expanded = np.expand_dims(pc1, axis=1)  # Shape (N, 1, 3)
    pc2_expanded = np.expand_dims(pc2, axis=0)  # Shape (1, M, 3)

    # Calculate squared distances between each pair of points
    distances = np.sqrt(np.sum((pc1_expanded - pc2_expanded) ** 2, axis=2))

    # Find the minimum distance for each point in pc1 to any point in pc2
    min_distances = np.min(distances, axis=1)

    # Calculate the average of these minimum distances
    avg_distance = np.mean(min_distances)

    return avg_distance


def hausdorff_distance(pc1, pc2):
    """
    Calculates the Hausdorff distance between two point clouds.

    Parameters:
    - pc1: numpy array of shape (N, 3), representing the first point cloud.
    - pc2: numpy array of shape (M, 3), representing the second point cloud.

    Returns:
    - hausdorff_dist: The Hausdorff distance between the two point clouds.
    """
    # Expand dimensions for broadcasting to calculate distances between each pair of points
    pc1_expanded = np.expand_dims(pc1, axis=1)  # Shape (N, 1, 3)
    pc2_expanded = np.expand_dims(pc2, axis=0)  # Shape (1, M, 3)

    # Calculate squared distances between each pair of points
    distances = np.sqrt(np.sum((pc1_expanded - pc2_expanded) ** 2, axis=2))

    # Find the minimum distance for each point in pc1 to any point in pc2
    min_distances_1_to_2 = np.min(distances, axis=1)

    # Find the minimum distance for each point in pc2 to any point in pc1
    min_distances_2_to_1 = np.min(distances, axis=0)

    # The Hausdorff distance is the maximum of these minimum distances
    hausdorff_dist = max(np.max(min_distances_1_to_2),
                         np.max(min_distances_2_to_1))

    return hausdorff_dist


def calculate_iqr(data):
    """
    Calculate the Inter-Quartile Range (IQR) and quartiles for a given dataset.

    Returns:
        tuple: (Q1, median, Q3, IQR)
    """
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return q1, median, q3, iqr


def process_stl_files(generated_dir, ground_truth_dir):
    """
    Process STL files, calculate metrics and print results directly.

    Parameters:
        generated_dir (str): Directory containing generated STL files
        ground_truth_dir (str): Directory containing ground truth STL files
    """
    # Initialize results list
    results = []

    # Get all ground truth files and extract base file names
    gt_files = list(Path(ground_truth_dir).glob("*_ground_truth.stl"))
    base_file_names = [f.stem.replace("_ground_truth", "") for f in gt_files]

    print(f"Found {len(base_file_names)} ground truth files")

    # Ensure we evaluate exactly 200 files (or however many are specified)
    total_files = 200
    processed_count = 0

    for base_name in base_file_names:
        if processed_count >= total_files:
            break
        pattern1 = os.path.join(generated_dir, f"{base_name}.stl")
        pattern2 = os.path.join(generated_dir, f"{base_name}_*.stl")
        matched_files = glob.glob(pattern1) + glob.glob(pattern2)
        print(f"Matched files for {base_name}: {matched_files}")
        if not matched_files:
            print(f"No generated file found for {base_name}")
            continue

        # latest modified file
        gen_path = max(matched_files, key=os.path.getmtime)
        gt_path = os.path.join(
            ground_truth_dir, f"{base_name}_ground_truth.stl")

        processed_count += 1
        print(f"Processing {base_name} ({processed_count}/{total_files})")

        # Load point clouds
        gen_pcd = stl_to_point_cloud(gen_path)
        gt_pcd = stl_to_point_cloud(gt_path)

        # Default values for failed compilations
        pc_dist = math.sqrt(3)
        haus_dist = math.sqrt(3)
        iogt = 0.0
        compilation_success = False

        if gen_pcd is not None and gt_pcd is not None:
            compilation_success = True
            # Apply ICP for alignment
            destination_points = np.asarray(gen_pcd.points)
            source_points = np.asarray(gt_pcd.points)

            destination_points = pc_normalize(destination_points)
            source_points = pc_normalize(source_points)
            T, finalA, final_error, i = iterative_closest_point(
                source_points, destination_points, 2000)
            iogt = intersection_over_GT2(destination_points, finalA)
            pc_dist = point_cloud_distance(destination_points, finalA)
            haus_dist = hausdorff_distance(destination_points, finalA)

        # Store results
        results.append([
            base_name,
            compilation_success,
            pc_dist,
            haus_dist,
            iogt
        ])

    # If we didn't get 200 files, pad with failed compilations
    while len(results) < total_files:
        missing_index = len(results) + 1
        results.append([
            f"missing_{missing_index}",
            False,
            math.sqrt(3),
            math.sqrt(3),
            0.0
        ])

    # Convert results to numpy array
    results_array = np.array(results, dtype=object)

    # Extract data for calculations
    compilation_success = results_array[:, 1].astype(bool)
    pc_dists = results_array[:, 2].astype(float)
    haus_dists = results_array[:, 3].astype(float)
    iogts = results_array[:, 4].astype(float)

    # Print basic summary
    print("\nBasic Summary:")
    print(f"Total files: {len(results_array)}")
    successful_count = np.sum(compilation_success)
    print(f"Successfully compiled: {successful_count}")
    print(f"Failed to compile: {len(results_array) - successful_count}")
    success_rate = successful_count / len(results_array)
    print(f"Compilation success rate: {success_rate:.4f}")

    # Process metrics for all files, not just successful ones
    metrics_to_analyze = ['point_cloud_distance', 'hausdorff_distance', 'iogt']
    metrics_data = [pc_dists, haus_dists, iogts]

    # Calculate statistics for all files (including failed compilations)
    print("\nStatistics for all files (including failed compilations):")
    for i, metric_name in enumerate(metrics_to_analyze):
        metric_data = metrics_data[i]

        # Calculate statistics using numpy methods
        mean_val = np.mean(metric_data)
        median_val = np.median(metric_data)
        q1 = np.percentile(metric_data, 25)
        q3 = np.percentile(metric_data, 75)
        iqr_val = q3 - q1
        min_val = np.min(metric_data)
        max_val = np.max(metric_data)

        # Print detailed statistics for all files
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(
            f"  Median: {median_val:.4f} (IQR: {iqr_val:.4f}, Q1: {q1:.4f}, Q3: {q3:.4f})")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  IQR: {iqr_val:.4f}")


def main():
    """
    Main function to run the evaluation.
    """
    # Set the directories
    generated_dir = "generated"
    ground_truth_dir = "data/Ground_truth"

    # Create directories if they don't exist
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)

    print("Starting evaluation of STL files...")
    print(f"Generated directory: {generated_dir}")
    print(f"Ground truth directory: {ground_truth_dir}")

    # Process STL files, calculate and print metrics all in one function
    process_stl_files(generated_dir, ground_truth_dir)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
