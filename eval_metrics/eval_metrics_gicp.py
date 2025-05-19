#Evalution script by us but not used for evaluation for consistent comparison between the two models
import numpy as np
import open3d as o3d
import glob
import os
import math
from pathlib import Path


def load_and_sample_stl(filename, sample_points=1000):
    try:
        mesh = o3d.io.read_triangle_mesh(filename)
        mesh.compute_vertex_normals()
        return mesh.sample_points_uniformly(number_of_points=sample_points)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def preprocess_pcd(pcd, voxel_size):
    # Downsample and estimate normals
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals()
    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # The mutual_filter parameter is added as the 5th argument (here set to False)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, 
        target_down,
        source_fpfh,
        target_fpfh,
        False,  # mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def normalize_point_cloud(pcd):
    """
    Normalize the point cloud to fit within a unit cube.
    """
    if pcd is None:
        return None
    
    # Get points as numpy array
    points = np.asarray(pcd.points)
    
    # Calculate min and max for each dimension
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # Calculate range for each dimension
    ranges = max_vals - min_vals
    max_range = np.max(ranges)
    
    # Scale points to fit in unit cube
    scaled_points = (points - min_vals) / max_range
    
    # Create new normalized point cloud
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(scaled_points)
    
    return normalized_pcd

def point_cloud_distance(pcd1, pcd2):
    """
    Calculate the point cloud distance between two point clouds.
    Returns sqrt(3) if either point cloud is None.
    """
    if pcd1 is None or pcd2 is None:
        return math.sqrt(3)
    
    # Calculate distances from pcd1 to pcd2
    distances = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    return np.mean(distances)

def hausdorff_distance(pcd1, pcd2):
    """
    Calculate the Hausdorff distance between two point clouds.
    Returns sqrt(3) if either point cloud is None.
    """
    if pcd1 is None or pcd2 is None:
        return math.sqrt(3)
    
    # Calculate distances from pcd1 to pcd2
    distances1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    
    # Calculate distances from pcd2 to pcd1
    distances2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
    # Hausdorff distance is the maximum of the two
    return max(np.max(distances1), np.max(distances2))

def compute_bounding_box(pcd):
    """
    Compute the axis-aligned bounding box for a point cloud.
    Returns None if the point cloud is None.
    """
    if pcd is None:
        return None
    
    # Get the axis-aligned bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    
    return (min_bound, max_bound)

def intersection_over_ground_truth(pcd1, pcd2):
    """
    Calculate the Intersection over Ground Truth (IoGT) between two point clouds.
    Returns 0 if either point cloud is None.
    """
    if pcd1 is None or pcd2 is None:
        return 0.0
    
    # Get bounding boxes
    bbox1 = compute_bounding_box(pcd1)
    bbox2 = compute_bounding_box(pcd2)
    
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    min_bound1, max_bound1 = bbox1
    min_bound2, max_bound2 = bbox2
    
    # Calculate intersection volume
    intersection_min = np.maximum(min_bound1, min_bound2)
    intersection_max = np.minimum(max_bound1, max_bound2)
    
    # Check if boxes overlap
    if np.any(intersection_min > intersection_max):
        return 0.0
    
    intersection_volume = np.prod(intersection_max - intersection_min)
    
    # Calculate ground truth volume
    gt_volume = np.prod(max_bound2 - min_bound2)
    
    if gt_volume == 0:
        return 0.0
    
    return intersection_volume / gt_volume

def run_icp(source, target, trans_init, threshold):
    """
    Apply the Iterative Closest Point algorithm to align source to target.
    Returns aligned source and transformation matrix.
    Returns source and identity matrix if either point cloud is None.
    """
    if source is None or target is None:
        # Return identity transformation if either point cloud is None
        return source, np.identity(4)
    
    # Make deep copies to avoid modifying the original point clouds
    source_copy = o3d.geometry.PointCloud(source)
    target_copy = o3d.geometry.PointCloud(target)
    
    # Initial alignment using point-to-point ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # Apply transformation to the source point cloud
    source_copy.transform(icp_result.transformation)
    
    return source_copy, icp_result.transformation

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

def evaluate_stl_files(generated_dir, ground_truth_dir):
    """
    Evaluate all generated STL files against their ground truth counterparts.
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
        
        gen_path = max(matched_files, key=os.path.getmtime)  # latest modified file
        gt_path = os.path.join(
            ground_truth_dir, f"{base_name}_ground_truth.stl")
        
        processed_count += 1
        print(f"Processing {base_name} ({processed_count}/{total_files})")
        
        # Load point clouds
        gen_pcd = load_and_sample_stl(gen_path)
        gt_pcd = load_and_sample_stl(gt_path)
        
        # Default values for failed compilations
        pc_dist = math.sqrt(3)
        haus_dist = math.sqrt(3)
        iogt = 0.0
        compilation_success = 0
        
        # Preprocess: downsample and compute features
        
        voxel_size = 0.05
        if gen_pcd is not None and gt_pcd is not None:
            source_down, source_fpfh = preprocess_pcd(gen_pcd, voxel_size)
            target_down, target_fpfh = preprocess_pcd(gt_pcd, voxel_size)

            # Global registration using RANSAC to get an initial alignment
            ransac_result = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        # Refine alignment with ICP so that the blue box overlaps the orange perfectly
        threshold = voxel_size * 0.5
        
        if gen_pcd is not None and gt_pcd is not None:
            compilation_success = True
            # Apply ICP for alignment
            aligned_gen_pcd, _ = run_icp(gen_pcd, gt_pcd,ransac_result.transformation,threshold)
            # Normalize point clouds

            gen_pcd_norm = normalize_point_cloud(aligned_gen_pcd)
            gt_pcd_norm = normalize_point_cloud(gt_pcd)
            
            
            
            # Calculate metrics
            pc_dist = point_cloud_distance(gen_pcd_norm, gt_pcd_norm)
            haus_dist = hausdorff_distance(gen_pcd_norm, gt_pcd_norm)
            iogt = intersection_over_ground_truth(gen_pcd_norm, gt_pcd_norm)
            compilation_success = 1
        
        
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
        print(f"  Median: {median_val:.4f} (IQR: {iqr_val:.4f}, Q1: {q1:.4f}, Q3: {q3:.4f})")
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
    evaluate_stl_files(generated_dir, ground_truth_dir)
    
    print("Evaluation complete.")


if __name__ == "__main__":
    main()