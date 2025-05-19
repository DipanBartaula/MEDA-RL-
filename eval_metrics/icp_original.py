'''Original ICP code from CADCodeVerify authors'''
import open3d as o3d
import numpy as np
import cv2
import os
from stl import mesh
from sklearn.neighbors import NearestNeighbors
import tqdm
from skimage.metrics import mean_squared_error, structural_similarity
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def stl_to_point_cloud(filename):
    point_cloud = o3d.io.read_triangle_mesh(filename)
    point_cloud = point_cloud.sample_points_poisson_disk(1000)

    return point_cloud

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# Credit for https://github.com/OmarJItani/Iterative-Closest-Point-Algorithm
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
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Translation
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    
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

    neigh = NearestNeighbors(n_neighbors=1)#n_neighbors=1
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def iterative_closest_point(A, B, max_iterations=20, tolerance=0.001):#tolerance=0.001

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
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):#tqdm.tqdm(range(max_iterations)):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error (stop if error is less than specified tolerance)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, error, and mapped source points
    T = best_fit_transform(A, src[:m,:].T)
    final_error = prev_error
    
    # get final A 
    rot = T[0:-1,0:-1]
    t = T[:-1,-1]
    finalA = np.dot(rot, A.T).T + t

    return T, finalA, final_error, i

def intersection_over_GT2(GT, test, threshold=-0.6):# -0.6
    # Convert to binary images based on a threshold
    #print(GT)
    GT_binary = (GT > threshold).astype(np.uint8)
    test_binary = (test > threshold).astype(np.uint8)
    #print(f"GT_binary:\n{GT_binary}")
    #print(f"test_binary:\n{test_binary}")
    # Compute the intersection of GT and test images
    intersection = np.logical_and(GT_binary, test_binary).astype(np.uint8)


    # Calculate the ratio of the intersection over GT's foreground
    if np.count_nonzero(GT_binary) == 0:
        return 0  # Handle division by zero if GT has no foreground pixels
    return np.count_nonzero(intersection) / np.count_nonzero(GT_binary)

def intersection_over_GT(GT, test):
    # Ensure both images are single-channel by converting them to grayscale if they are not
    if len(GT.shape) == 3:
        GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
    if len(test.shape) == 3:
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    #intersection = cv2.bitwise_and(GT, test)
    intersection = np.logical_and(GT, test).astype(np.uint8)

    return cv2.countNonZero(intersection) / cv2.countNonZero(GT)

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

def voxel_grid2(point_cloud, grid_size):
    # Normalizing coordinates to the grid size
    min_vals = np.min(point_cloud, axis=0)
    max_vals = np.max(point_cloud, axis=0)
    # Adjust max_vals slightly to ensure points on the upper boundary are included
    max_vals += 1e-9
    scales = (grid_size - 1) / (max_vals - min_vals)
    voxels = np.floor((point_cloud - min_vals) * scales).astype(int)
    # Ensure all voxel indices are within the grid size
    voxels = np.clip(voxels, 0, grid_size - 1)
    return set(map(tuple, voxels))

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
    hausdorff_dist = max(np.max(min_distances_1_to_2), np.max(min_distances_2_to_1))

    return hausdorff_dist

def iou(y_true, y_pred):
 
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    print(iou_score)
    return iou_score

def calculate_iqr(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the IQR
    IQR = Q3 - Q1
    
    return IQR


folder_path = "//Users/kamel/Desktop/Self_verification/Resutls/Finals/Few_shot_GPT4_Subset_of_50_Human_and_Machine"

items = os.listdir(folder_path)
folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]



STLs = [ 'Generated.stl', 'Premise_Refine_1.stl', 'Premise_Refine_2.stl', 'self-verification_Refine_1.stl', 'self-verification_Refine_2.stl',
        'GS_Refine_1.stl', 'GS_Refine_2.stl' ]

STLs = ['Generated.stl', 'Human_and_Machine_Refine_1.stl', 'Human_and_Machine_Refine_2.stl']

complexity_dict = {'00000007': 1, '00000633': 2, '00000960': 2, '00001411': 1, '00001490': 2, '00001615': 3, '00001817': 3, '00001977': 2, '00002221': 3, '00002298': 2, '00003219': 1, '00003247': 1, '00003558': 1, '00003763': 3, '00003801': 2, '00004154': 2, '00004495': 2, '00004596': 2, '00004935': 3, '00005161': 3, '00005358': 2, '00005721': 1, '00006013': 1, '00006136': 1, '00006578': 3, '00006863': 1, '00006892': 3, '00007258': 1, '00007362': 1, '00008138': 2, '00008315': 1, '00008597': 1, '00008835': 1, '00009044': 2, '00009307': 3, '00009529': 3, '00009823': 1, '00009843': 3, '00009863': 1, '00009998': 3, '00017291': 3, '00019015': 3, '00019066': 3, '00031181': 3, '00031303': 4, '00031637': 3, '00032961': 3, '00033093': 3, '00033624': 3, '00034100': 4, '00034239': 3, '00034243': 3, '00034256': 4, '00036518': 4, '00037135': 4, '00037161': 2, '00037276': 3, '00037494': 2, '00038438': 2, '00038614': 2, '00039012': 3, '00039227': 3, '00039365': 3, '00039681': 4, '00039777': 3, '00300037': 4, '00520130': 3, '00520150': 3, '00520321': 3, '00520402': 4, '00520453': 2, '00520570': 2, '00520638': 4, '00520671': 4, '00520675': 2, '00520699': 4, '00520726': 3, '00520776': 2, '00520974': 3, '00520976': 3, '00521000': 3, '00521025': 3, '00521217': 2, '00521230': 4, '00521437': 3, '00521895': 3, '00521969': 2, '00522355': 3, '00522404': 3, '00522865': 4, '00523882': 4, '00524912': 3, '00670105': 2, '00670106': 1, '00670231': 3, '00670256': 2, '00670259': 2, '00670266': 3, '00670268': 4, '00670274': 3, '00670279': 2, '00670334': 3, '00670441': 2, '00670454': 3, '00670457': 3, '00670466': 2, '00670817': 2, '00670960': 4, '00671873': 3, '00671898': 4, '00671938': 4, '00672098': 3, '00672272': 4, '00672291': 4, '00672309': 4, '00672352': 3, '00672355': 3, '00672359': 4, '00672804': 3, '00673733': 3, '00673788': 3, '00675092': 2, '00675498': 3, '00675952': 3, '00676218': 2, '00680715': 2, '00681053': 4, '00681399': 4, '00681463': 2, '00681547': 3, '00681589': 4, '00681754': 3, '00681760': 4, '00681831': 4, '00681999': 3, '00682073': 3, '00682075': 4, '00684686': 3, '00684841': 4, '00685823': 3, '00689273': 4, '00689964': 3, '00851553': 4, '00852000': 3, '00853706': 4, '00857821': 4, '00980412': 4, '00980651': 3, '00982481': 4, '00983173': 4, '00984033': 4, '00984234': 3, '00984488': 2, '00984833': 2, '00985066': 3, '00985482': 3, '00985494': 3, '00986712': 3, '00986814': 3, '00995686': 4, '00995733': 4, '00995759': 3, '00995843': 2, '00996001': 3, '00996329': 2, '00996368': 4, '00996457': 4, '00996473': 3, '00996962': 4, '00997040': 4, '00997065': 3, '00997068': 4, '00997071': 4, '00997229': 4, '00997300': 3, '00997373': 3, '00997428': 4, '00997536': 3, '00997580': 2, '00997616': 3, '00997677': 3, '00997681': 3, '00997753': 3, '00997785': 4, '00997852': 2, '00997878': 3, '00998012': 4, '00998074': 4, '00998088': 4, '00998283': 3, '00998300': 3, '00998356': 4, '00998398': 3, '00998698': 4, '00998714': 4, '00998749': 3, '00998843': 4, '00999126': 4, '00999141': 3, '00999374': 4}

count = 0
for STL in STLs:
    Resutls_ICP = []
    final_error_ICP = []
    Res_hausdorff_distance = []
    Res_point_cloud_distance = []
    count_non_complie = 0 
    for folder in folders:#tqdm.tqdm(folders):
        #print(complexity_dict[folder])
        #if complexity_dict[folder] not in [4]:
        #    continue
        #if folder != '00001411':
        #    continue
        if not os.path.exists(os.path.join(folder_path, folder, STL)):
            #print(source_points)
            Resutls_ICP.append(0)
            Res_point_cloud_distance.append(1.73)
            Res_hausdorff_distance.append(1.73)

            count_non_complie = count_non_complie + 1
            #print(folder, ',', 1.73)
            #print(folder, ',', 1.73)
            continue

        destination_points = stl_to_point_cloud(os.path.join(folder_path, folder, "Ground_Truth.stl"))
        source_points = stl_to_point_cloud(os.path.join(folder_path, folder, STL))


        destination_points = np.asarray(destination_points.points)
        source_points = np.asarray(source_points.points)

        destination_points = pc_normalize(destination_points)
        source_points= pc_normalize(source_points)

        


        T, finalA, final_error, i = iterative_closest_point(source_points, destination_points, 2000)


        #Resutls.append(intersection_over_GT2(destination_points, source_points))
     
        IoGT = intersection_over_GT2(destination_points, finalA)
        
        Resutls_ICP.append(IoGT)

        
        distance = point_cloud_distance(destination_points, finalA)
        Res_point_cloud_distance.append(distance)
        #print(folder, ',', round(distance,3))
        #print(folder, ': ', distance)

        distance = hausdorff_distance(destination_points, finalA)
        Res_hausdorff_distance.append(distance)
        #print(folder, ',', round(distance, 3))
        final_error_ICP.append(final_error)
        #print(folder, ',', round(distance,3))


    
    count_non_complie = len(Res_point_cloud_distance) - len([x for x in Res_point_cloud_distance if x != 1.73])

    print(STL)
    #print(len(Res_point_cloud_distance))
    #print(f"Mean: IoGT: {round(np.mean(Resutls_ICP),3)}, PLC distances: {round(np.mean(Res_point_cloud_distance),3)}({round(np.std(Res_point_cloud_distance),3)}), hausdorff distance: {round(np.mean(Res_hausdorff_distance),3)}({round(np.std(Res_hausdorff_distance),3)}), Success Rate: {((200 - count_non_complie))/200 * 100}")
    print(f"Median: IoGT: {round(np.median(Resutls_ICP),3)}({round(calculate_iqr(Resutls_ICP),3)}), PLC distances: {round(np.median(Res_point_cloud_distance),3)}({round(calculate_iqr(Res_point_cloud_distance),3)}), hausdorff distance: {round(np.median(Res_hausdorff_distance),3)}({round(calculate_iqr(Res_hausdorff_distance),3)}), Success Rate: {((len(Res_point_cloud_distance) - count_non_complie))/len(Res_point_cloud_distance) * 100}")
    #print(np.mean(Resutls))
    #print(f"Success Rate{((200 - count_non_complie))/200 * 100}")
#print(Resutls_ICP)
