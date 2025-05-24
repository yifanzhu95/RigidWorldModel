import math
from math import exp
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from icecream import ic
import open3d as o3d

def convert_camera_pytorch3D(id, view_matrix, projectionMatrix, image, data_device, near, far, fov):
    v = view_matrix
    R = np.array([[v[0],v[4],v[8]],\
            [v[1],v[5],v[9]],\
            [v[2],v[6],v[10]]])
    t = np.array(v[12:15]) #world to cam
    # ic(R,t)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] =  t
    T = np.linalg.inv(T) # cam to world
    T[:3,[0,2]] = -T[:3,[0,2]] #convert camera convention (opengl -> pytorch3d)
    
    T_world_to_cam = np.linalg.inv(T) 
    world_to_cam_t = T_world_to_cam[:3,3]
    # height, width, _ = image.shape
    p = projectionMatrix
    projectionMatrix = np.array([[p[0],p[4],p[8],p[12]],\
            [p[1],p[5],p[9],p[13]],\
            [p[2],p[6],p[10],p[14]],\
            [p[3],p[7],p[11],p[15]]])

    # the pytorch3d convention is atrocious -- cam to world rotation, but world to cam translation..
    R = torch.from_numpy(T[:3,:3]).float().unsqueeze(0).to(data_device)
    T = torch.from_numpy(world_to_cam_t).float().unsqueeze(0).to(data_device)

    cam = FoVPerspectiveCameras(R = R,
        T = T,
        device = data_device,
        znear=near,
        zfar=far,
        aspect_ratio=1,
        fov = fov, #45.0,
        degrees=True)

    return cam

def convert_camera_pytorch3D_from_camM(view_matrix, camera_matrix, image_height, image_width, data_device):
    v = view_matrix
    R = np.array([[v[0],v[4],v[8]],\
            [v[1],v[5],v[9]],\
            [v[2],v[6],v[10]]])
    t = np.array(v[12:15]) #world to cam
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] =  t
    T = np.linalg.inv(T) # cam to world
    #T[:3,[0,2]] = -T[:3,[0,2]] #convert camera convention
    
    T_world_to_cam = np.linalg.inv(T) 
    world_to_cam_t = T_world_to_cam[:3,3]

    # the pytorch3d convention is atrocious -- cam to world rotation, but world to cam translation..
    R = torch.from_numpy(T[:3,:3]).float().unsqueeze(0).to(data_device)
    T = torch.from_numpy(world_to_cam_t).float().unsqueeze(0).to(data_device)

    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    px = camera_matrix[0,2]
    py = camera_matrix[1,2]
    f = torch.from_numpy(np.array([fx,fy])).unsqueeze(0).float()
    principle_pt = torch.from_numpy(np.array([px,py])).unsqueeze(0).float()
    sizes = torch.from_numpy(np.array([image_height,image_width])).unsqueeze(0).float()
    cam = PerspectiveCameras(
        R = R,
        T = T,
        device = data_device,
        focal_length=f,
        principal_point=principle_pt,
        image_size=sizes,
        in_ndc=False)
    return cam


def compute_pcd(rgb, depth, view_matrix, proj_matrix):
    # https://github.com/bulletphysics/bullet3/issues/1924
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")

    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
    height, width, _ = rgb.shape
    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 15] #pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, final = 0.00001):
        self.initial = initial
        self.interval = interval
        self.factor = factor
        self.final = final

    def get_learning_rate(self, epoch):

        return max(self.final, self.initial * (self.factor ** (epoch // self.interval)))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length

def get_learning_rate_schedules(specs, name):

    schedule_specs = specs[name]

    schedules = []
    
    for schedule_specs in schedule_specs:
        final = schedule_specs.get('Final', 0.00001)
        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def adjust_learning_rate(lr_schedules, optimizer, epoch):

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

mse = nn.MSELoss()

def triangle_point_distance_and_normal(point, triangle_vertices, vertex_normals):
    """
    Calculate the closest point on a triangle to a given point in 3D space.
    
    Args:
    point (torch.Tensor): The point in 3D space (shape: [3])
    triangle_vertices (torch.Tensor): The vertices of the triangle (shape: [3, 3])
    vertex_normals (torch.Tensor): The normals at each vertex of the triangle (shape: [3, 3])
    
    Returns:
    tuple: (closest_point, closest_point_normal, distance)
        closest_point (torch.Tensor): The closest point on the triangle (shape: [3])
        closest_point_normal (torch.Tensor): The interpolated normal at the closest point (shape: [3])
        distance (torch.Tensor): The distance between the query point and the closest point (scalar)
    """
    # Compute edge vectors
    edge0 = triangle_vertices[1] - triangle_vertices[0]
    edge1 = triangle_vertices[2] - triangle_vertices[0]
    
    # Compute the vector from a triangle vertex to the point
    v = point - triangle_vertices[0]
    
    # Compute dot products
    d00 = torch.dot(edge0, edge0)
    d01 = torch.dot(edge0, edge1)
    d11 = torch.dot(edge1, edge1)
    d20 = torch.dot(v, edge0)
    d21 = torch.dot(v, edge1)
    
    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Clamp barycentric coordinates to [0, 1]
    u = torch.clamp(u, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    
    # Normalize barycentric coordinates
    total = u + v + w
    u /= total
    v /= total
    w /= total
    
    # Compute the closest point on the triangle
    closest_point = u * triangle_vertices[0] + v * triangle_vertices[1] + w * triangle_vertices[2]
    
    # Interpolate the normal at the closest point
    closest_point_normal = u * vertex_normals[0] + v * vertex_normals[1] + w * vertex_normals[2]
    closest_point_normal = closest_point_normal / torch.norm(closest_point_normal)
    
    # Calculate the distance between the query point and the closest point
    distance = torch.norm(point - closest_point)
    
    return closest_point, closest_point_normal, distance


def triangle_point_distance_and_normal_batched(points, triangle_vertices, vertex_normals, eps = 1e-4):
    """
    Calculate the closest points on triangles to given points in 3D space in batch,
    along with the distances and interpolated normals.
    
    Args:
    points (torch.Tensor): The points in 3D space (shape: [B, 3])
    triangle_vertices (torch.Tensor): The vertices of the triangles (shape: [B, 3, 3])
    vertex_normals (torch.Tensor): The normals at each vertex of the triangles (shape: [B, 3, 3])
    
    Returns:
    tuple: (closest_points, closest_point_normals, distances)
        closest_points (torch.Tensor): The closest points on the triangles (shape: [B, 3])
        closest_point_normals (torch.Tensor): The interpolated normals at the closest points (shape: [B, 3])
        distances (torch.Tensor): The distances between the query points and the closest points (shape: [B])
    """
    # Compute edge vectors
    edge0 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge1 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    
    # Compute the vectors from triangle vertices to the points
    v = points - triangle_vertices[:, 0]
    
    # Compute dot products
    d00 = torch.sum(edge0 * edge0, dim=1)
    d01 = torch.sum(edge0 * edge1, dim=1)
    d11 = torch.sum(edge1 * edge1, dim=1)
    d20 = torch.sum(v * edge0, dim=1)
    d21 = torch.sum(v * edge1, dim=1)
    
    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Clamp barycentric coordinates to [0, 1]
    u = torch.clamp(u, 0.0, 1.0).unsqueeze(1)
    v = torch.clamp(v, 0.0, 1.0).unsqueeze(1)
    w = torch.clamp(w, 0.0, 1.0).unsqueeze(1)
    
    # Normalize barycentric coordinates
    total = u + v + w
    u = u / total
    v = v / total
    w = w / total
    
    # Compute the closest points on the triangles
    closest_points = u * triangle_vertices[:, 0] + v * triangle_vertices[:, 1] + w * triangle_vertices[:, 2]
    
    # Interpolate the normals at the closest points
    closest_point_normals = u * vertex_normals[:, 0] + v * vertex_normals[:, 1] + w * vertex_normals[:, 2]
    closest_point_normals = closest_point_normals / torch.norm(closest_point_normals, dim=1, keepdim=True)
   
    # Calculate the vector from closest point to query point
    point_to_surface = points - closest_points
    
    # Calculate the unsigned distances
    unsigned_distances = torch.norm(torch.abs(point_to_surface) + eps, dim=1)
    
    # Calculate the sign using dot product between point_to_surface and normal
    # If dot product is positive, point is in direction of normal (outside)
    # If dot product is negative, point is opposite to normal (inside)
    signs = torch.sign(torch.sum(point_to_surface * closest_point_normals, dim=1))
    
    # Calculate signed distances
    signed_distances = signs * unsigned_distances
    
    # Check for NaN values
    isnan = torch.isnan(signed_distances)
    
    return closest_points, closest_point_normals, signed_distances, isnan

def get_discrete_sphere(r):
    thetas = np.arange(0, 360, 10)/180*math.pi
    phis = np.linspace(0, 180, 31)/180*math.pi
    points = []
    for theta in thetas:
        for phi in phis: 
            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta) * r
            y = np.sin(phi) * np.sin(theta) * r
            z = np.cos(phi) * r
            points.append([x,y,z])
    points = [list(t) for t in set(tuple(inner) for inner in points)]
    return np.array(points)


def register_pcd(source, target, voxel_size = 0.005):
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    def draw_registration_result(source, target, transformation, scale=1.0):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.scale(scale, center=source_temp.get_center())
        source_temp.transform(transformation)
        try:
            o3d.visualization.draw_geometries([source_temp, target_temp])
        except Exception as e:
            print(f"An error occurred: {e}")

    def icp_with_scale(source, target, init_translation, init_rotation, init_scale):
        def objective_func(x):
            scale = x[0]
            translation = x[1:4]
            rotation = Rotation.from_rotvec(x[4:7]).as_matrix()
            
            scaled_source = copy.deepcopy(source)
            scaled_source.scale(scale, center=scaled_source.get_center())
            
            transformation = np.eye(4)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = translation
            
            scaled_source.transform(transformation)
            
            #distances = np.asarray(scaled_source.compute_point_cloud_distance(target))
            distances = np.asarray(target.compute_point_cloud_distance(scaled_source))
            return np.mean(distances)

        x0 = np.concatenate([[init_scale], init_translation, init_rotation])
        result = minimize(objective_func, x0, bounds = [(0.5,1.5),(-2,2),(-2,2),(-2,2),\
                            (-3,3),(-3,3),(-3,3)]) #, method='Powell'
        
        optimal_scale = result.x[0]
        optimal_translation = result.x[1:4]
        optimal_rotation = Rotation.from_rotvec(result.x[4:7]).as_matrix()
        
        optimal_transformation = np.eye(4)
        optimal_transformation[:3, :3] = optimal_rotation
        optimal_transformation[:3, 3] = optimal_translation
        
        return optimal_transformation, optimal_scale
    
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    if len(source.points) == 0 or len(target.points) == 0:
        print("Error: One or both point clouds are empty.")
        exit()

    # prepare the point clouds
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    try:
        o3d.visualization.draw_geometries([source_down, target_down], window_name = 'Downsampled pcds for registration')
    except Exception as e:
        print(f"An error occurred: {e}")
    # Prepare RANSAC
    distance_threshold = 1.5*voxel_size
    print(":: RANSAC registration on downsampled point clouds.")
    print("   distance threshold %.3f." % distance_threshold)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(40000000, 500))


    ransac_res_transformation = np.array(result_ransac.transformation)
    source_down.transform(ransac_res_transformation)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    try:
        o3d.visualization.draw_geometries([source_down, target_down, frame], window_name = 'RANSAC result')
    except Exception as e:
        print(f"An error occurred during open3d visualization: {e}")
    
    final_transformation = ransac_res_transformation

    # Refine the result using ICP, and optimize for scale
    init_transformation = ransac_res_transformation #np.array(result_ransac.transformation)  # Create a copy
    init_translation = init_transformation[:3, 3]
    init_rotation = Rotation.from_matrix(init_transformation[:3, :3]).as_rotvec()
    init_scale = 1.0
    print(":: Scale-aware ICP registration is applied on original point clouds.")
    final_transformation, final_scale = icp_with_scale(source, target, init_translation, init_rotation, init_scale)

    print("Final Transformation is:")
    print(final_transformation)
    print("Final Scale is:", final_scale)
    return final_transformation, final_scale


def register_pcd_ransac(source, target, voxel_size = 0.005):
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    if len(source.points) == 0 or len(target.points) == 0:
        print("Error: One or both point clouds are empty.")
        exit()

    # prepare the point clouds
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    try:
        o3d.visualization.draw_geometries([source_down, target_down], window_name = 'Downsampled pcds for registration')
    except Exception as e:
        print(f"An error occurred: {e}")
    # Prepare RANSAC
    distance_threshold = 2*voxel_size
    print(":: RANSAC registration on downsampled point clouds.")
    print("   distance threshold %.3f." % distance_threshold)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(40000000, 500))
    
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(40000000, 500))

    #  
    source_down.transform(result_ransac.transformation)
    try:
        o3d.visualization.draw_geometries([source_down, target_down], window_name = 'RANSAC result')
    except Exception as e:
        print(f"An error occurred: {e}")
    transformation = np.array(result_ransac.transformation)  

    return transformation

def register_pcd_icp(source, target, max_iterations=30, threshold = 0.01):
    # Initialize parameters for ICP
    current_transformation = np.identity(4)
    
    # Create ICP registration object
    icp = o3d.pipelines.registration.registration_icp(
        source, target,
        threshold,
        current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    # Get the transformation matrix
    transformation_matrix = icp.transformation
    
    return transformation_matrix


def headless_open3d(meshes, width=640, height=480):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for mesh in meshes:
        vis.add_geometry(mesh)

        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
    
    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return np.asarray(image)

class PartialChamferLoss(nn.Module):
    def __init__(self, num_samples=1000):
        super(PartialChamferLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, pred, target):
        """
        Compute the partial unidirectional Chamfer loss for large point clouds
        
        Args:
        pred (torch.Tensor): Predicted point cloud (B, N, 3)
        target (torch.Tensor): Target point cloud (B, M, 3)
        
        Returns:
        torch.Tensor: Partial Chamfer loss
        """
        assert pred.dim() == target.dim() == 3, "Input tensors must be 3-dimensional"
        assert pred.size(0) == target.size(0), "Batch sizes must match"

        B, N, _ = pred.size()
        M = target.size(1)

        # Randomly sample points from pred if N > num_samples
        if N > self.num_samples:
            idx = torch.randperm(N, device=pred.device)[:self.num_samples]
            pred_sampled = pred.gather(1, idx.view(1, -1, 1).expand(B, -1, 3))
        else:
            pred_sampled = pred

        # Compute pairwise distances for all sampled points at once
        dist = torch.cdist(pred_sampled, target, p=2)
        
        # Find the minimum distance for each sampled point
        min_dist, _ = torch.min(dist, dim=2)
        
        # Compute the mean loss across all batches and sampled points
        loss = torch.mean(min_dist)

        return loss


class CustomChamferLoss(nn.Module):
    """
    Chamfer Distance loss implemented as a PyTorch module with memory optimization.
    Limits the maximum number of points used to save memory for large point clouds.
    """
    
    def __init__(self, max_points=None, reduction='mean', chunk_size=None):
        """
        Initialize the Chamfer Distance loss module.
        
        Args:
            max_points: Maximum number of points to use from each point cloud. 
                        If None, use all points. If int, randomly sample that many points.
            reduction: 'mean', 'sum' or 'none'. Default: 'mean'
                'none': no reduction will be applied
                'mean': the mean of the loss
                'sum': the sum of the loss
            chunk_size: If not None, compute pairwise distances in chunks to save memory.
                        Useful for very large point clouds.
        """
        super(CustomChamferLoss, self).__init__()
        self.max_points = max_points
        self.reduction = reduction
        self.chunk_size = chunk_size
    
    def _subsample_points(self, x, max_points):
        """Randomly subsample points if there are more than max_points."""
        batch_size, n_points, dim = x.shape
        if n_points <= max_points:
            return x
            
        # Create indices for random sampling
        idx = torch.randperm(n_points, device=x.device)[:max_points]
        return x[:, idx, :]
    
    def _compute_chunked(self, x, y):
        """Compute pairwise distances in chunks to save memory."""
        batch_size, n_x, dim = x.shape
        _, n_y, _ = y.shape
        
        dist_x_to_y = torch.zeros(batch_size, n_x, device=x.device)
        dist_y_to_x = torch.zeros(batch_size, n_y, device=x.device)
        
        # Process x points in chunks
        for i in range(0, n_x, self.chunk_size):
            end_i = min(i + self.chunk_size, n_x)
            x_chunk = x[:, i:end_i, :].unsqueeze(2)  # (batch_size, chunk, 1, dim)
            
            # Process y points in chunks for each x chunk
            min_dists = torch.full((batch_size, end_i - i), float('inf'), device=x.device)
            
            for j in range(0, n_y, self.chunk_size):
                end_j = min(j + self.chunk_size, n_y)
                y_chunk = y[:, j:end_j, :].unsqueeze(1)  # (batch_size, 1, chunk, dim)
                
                # Compute distances between this chunk of x and y
                chunk_dist = torch.sum((x_chunk - y_chunk) ** 2, dim=-1)  # (batch_size, x_chunk, y_chunk)
                
                # Update minimum distances for x to y
                min_dists = torch.min(min_dists, torch.min(chunk_dist, dim=2)[0])
                
                # Update y to x at appropriate indices
                y_to_x_chunk_min = torch.min(chunk_dist, dim=1)[0]  # (batch_size, y_chunk)
                dist_y_to_x[:, j:end_j] = torch.min(
                    dist_y_to_x[:, j:end_j],
                    y_to_x_chunk_min if j == 0 else torch.min(dist_y_to_x[:, j:end_j], y_to_x_chunk_min)
                )
            
            # Store the computed minimum distances for this x chunk
            dist_x_to_y[:, i:end_i] = min_dists
            
        return dist_x_to_y, dist_y_to_x
    
    def _compute_full(self, x, y):
        """Compute all pairwise distances at once."""
        # Reshape for pairwise distance computation
        xx = x.unsqueeze(2)  # (batch_size, n_x, 1, dim)
        yy = y.unsqueeze(1)  # (batch_size, 1, n_y, dim)
        
        # Compute squared Euclidean distances
        dist = torch.sum((xx - yy) ** 2, dim=-1)  # (batch_size, n_x, n_y)
        
        # Get minimum distance for each point
        dist_x_to_y = torch.min(dist, dim=2)[0]  # (batch_size, n_x)
        dist_y_to_x = torch.min(dist, dim=1)[0]  # (batch_size, n_y)
        
        return dist_x_to_y, dist_y_to_x
    
    def forward(self, x, y):
        """
        Forward pass to compute the Chamfer distance.
        
        Args:
            x: (batch_size, num_points_x, point_dim) tensor of points
            y: (batch_size, num_points_y, point_dim) tensor of points
            
        Returns:
            Chamfer distance between the point clouds
        """
        # Apply point subsampling if needed
        if self.max_points is not None:
            x = self._subsample_points(x, self.max_points)
            y = self._subsample_points(y, self.max_points)
        
        # Compute pairwise distances (either in chunks or all at once)
        if self.chunk_size is not None:
            dist_x_to_y, dist_y_to_x = self._compute_chunked(x, y)
        else:
            dist_x_to_y, dist_y_to_x = self._compute_full(x, y)
        
        # Apply reduction
        if self.reduction == 'mean':
            chamfer_x = torch.mean(dist_x_to_y, dim=1)  # (batch_size)
            chamfer_y = torch.mean(dist_y_to_x, dim=1)  # (batch_size)
            chamfer = torch.mean(chamfer_x + chamfer_y)  # scalar
        elif self.reduction == 'sum':
            chamfer_x = torch.sum(dist_x_to_y, dim=1)  # (batch_size)
            chamfer_y = torch.sum(dist_y_to_x, dim=1)  # (batch_size)
            chamfer = torch.sum(chamfer_x + chamfer_y)  # scalar
        else:  # 'none'
            chamfer_x = dist_x_to_y  # (batch_size, n_x)
            chamfer_y = dist_y_to_x  # (batch_size, n_y)
            chamfer = (chamfer_x, chamfer_y)  # tuple of two tensors
        
        return chamfer

def quaternion_conjugate(q):
    # Input shape: (..., 4) where q = [x, y, z, w]
    q_conj = q.clone()
    q_conj[..., :3] = -q_conj[..., :3]
    return q_conj

def quaternion_multiply(q1, q2):
    # Multiply two quaternions q1 * q2
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((x, y, z, w), dim=-1)

def quaternion_geodesic_distance(q1, q2, eps=1e-7):
    # Ensure unit quaternions
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + eps)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + eps)

    dot_product = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(torch.abs(dot_product))
    return angle  # in radians

def quaternion_angle(q, eps=1e-7):
    # Input q: (..., 4), assumed to be [x, y, z, w]
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(q[..., 3]), -1.0, 1.0))  # w is last
    return angle

def quaternion_difference(q1, q2):
    # Returns the quaternion delta_q such that: q2 ≈ delta_q * q1
    q1_conj = quaternion_conjugate(q1)
    return quaternion_angle(quaternion_multiply(q2, q1_conj))

if __name__=='__main__':
    get_discrete_sphere(1)
    