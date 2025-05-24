
import copy
import math
from pathlib import Path    
import os

import numpy as np
import zarr 
from scipy.spatial import KDTree
import cv2
import open3d as o3d
from klampt.math import se3

from diffworld.utils import register_pcd, compute_pcd

class dataloader:
    def __init__(self, fn, configs) -> None:
        self.configs = configs
        path = Path(fn)
        self.root = zarr.open(str(path), mode='r')
        self.num_images = copy.copy(self.root.attrs['numImages'])
        self.objects = copy.copy(self.root.attrs['objects'])
        self.robotIndex = -1
        self.objIndex = -1
        for b in self.objects:
            if b[1] == 'robot':
                self.robotIndex = b[0]
            if b[1] == 'obj':
                self.objIndex = b[0]
        try:
            self.num_objs = len(self.objects)
            self.projection_matrix = copy.copy(self.root.attrs['projectionMatrix'])
            self.fovx = copy.copy(self.root.attrs['fovx'])
            self.fovy =copy.copy(self.root.attrs['fovy'])
        except:
            print('Zarr does not contain all attributes')

        if 'far' in self.root.attrs:
            self.far = copy.copy(self.root.attrs['far'])
            self.near =copy.copy(self.root.attrs['near'])
        else:
            self.far= 2.
            self.near = 0.1

        if 'cameraMatrix' in self.root.attrs:
            self.camera_matrix = np.array(copy.copy(self.root.attrs['cameraMatrix']))
        else:
            self.camera_matrix = None
        rgbd_source = self.configs.get('rgbd_source', 'pybullet')
        self.depth_scale = 1
        if rgbd_source == 'realsense':
            self.depth_scale = 1/1000
    def __len__(self):
        return self.num_images

    def get_num_objects(self):
        return self.num_objs
    
    def get_projection_matrix(self):
        return copy.deepcopy(self.projection_matrix)
    
    def get_fovs(self):
        return self.fovx, self.fovy

    def get_objects(self):
        return copy.deepcopy(self.objects)

    def get_near_far_planes(self):
        return self.near, self.far

    def get_camera_matrix(self):
        return copy.deepcopy(self.camera_matrix)
    
class DataloaderStage1ImageOnly(dataloader):
    def __init__(self, folder, zarr_name, configs) -> None:
        super().__init__(os.path.join(folder, zarr_name), configs)
        self.folder = folder
        self.indices = list(np.arange(0, self.num_images))

    def __len__(self):
        return self.num_images #len(self.indices)

    def __getitem__(self, i):
        return copy.copy(self.root['rgb_images'][i,:])/255.,\
            copy.copy(self.root['depth_images'][i,:])*self.depth_scale,\
            copy.copy(self.root['masks'][i,:]),\
            copy.copy(self.root['poses'][i,:])

    def get_all_data(self):
        return copy.copy(self.root['rgb_images'][:])/255.,\
            copy.copy(self.root['depth_images'][:])*self.depth_scale,\
            copy.copy(self.root['masks'][:]),\
            copy.copy(self.root['poses'][:])

class DataloaderStage1(dataloader):
    def __init__(self, folder, zarr_name, configs) -> None:
        super().__init__(os.path.join(folder, zarr_name), configs)
        self.folder = folder
        self.indices = [0] #only keep the first image
        self._compute_initial_pcd()
        
    def __len__(self):
        return self.num_images 

    def __getitem__(self, i):
        return copy.copy(self.root['rgb_images'][i,:])/255.,\
            copy.copy(self.root['depth_images'][i,:])*self.depth_scale,\
            copy.copy(self.root['masks'][i,:]),\
            copy.copy(self.root['poses'][i,:])

    def get_all_data(self):
        return copy.copy(self.root['rgb_images'][:])/255.,\
            copy.copy(self.root['depth_images'][:])*self.depth_scale,\
            copy.copy(self.root['masks'][:]),\
            copy.copy(self.root['poses'][:])

    def get_pcd(self):
        return copy.copy(self.pcds), copy.copy(self.pcd_mask),\
             copy.copy(self.visible_pcds), copy.copy(self.visible_pcd_mask)

    def _compute_initial_pcd(self):
        """
        Precompute the initial point cloud of the objects
        """
        pcd_fn = os.path.join(self.folder, 'pcds.npy')
        pcd_mask_fn = os.path.join(self.folder, 'pcd_masks.npy')
        visible_pcd_fn = os.path.join(self.folder, 'visible_pcds.npy')
        visible_pcd_mask_fn = os.path.join(self.folder, 'visible_pcd_masks.npy')

        # load if already computed
        if os.path.exists(pcd_fn):
            self.pcds = np.load(pcd_fn)
            self.pcd_mask = np.load(pcd_mask_fn)
            self.visible_pcds = np.load(visible_pcd_fn)
            self.visible_pcd_mask = np.load(visible_pcd_mask_fn)
            return 
        
        xyzs_all_objects = [[] for i in range(self.get_num_objects())]
        rgbs_all_objects = [[] for i in range(self.get_num_objects())]
        normals_all_objects = [None for i in range(self.get_num_objects())]
        rgbd_source = self.configs.get('rgbd_source', 'pybullet')

        # load and process all frames from the sequence
        for i in range(self.num_images):
            print(i/self.num_images)
            rgb, depth, mask, pose = self.__getitem__(i)
            xyzs = compute_pcd(rgb, depth, pose, self.get_projection_matrix())
            rgbs = rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]))
            mask = mask.reshape(-1)
            # compute normal and orient
            view_matrix = pose
            camera_T = se3.inv((list(view_matrix[0:3]) + list(view_matrix[4:7]) + list(view_matrix[8:11]), \
                        list(view_matrix[12:15])))
            camera_t = camera_T[1]
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(xyzs)
            o3d_pcd.colors = o3d.utility.Vector3dVector(rgbs)
            o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d_pcd.orient_normals_towards_camera_location(camera_t)
            normals = np.asarray(o3d_pcd.normals)
            for j in range(self.get_num_objects()):
                if j == self.robotIndex:
                    continue
                if j == self.objIndex and i > 0:
                    continue
                xyzs_all_objects[j] += list(xyzs[mask == j])
                rgbs_all_objects[j] += list(rgbs[mask == j])
                if normals_all_objects[j] is None:
                    normals_all_objects[j] = normals[mask == j]
                else:
                    normals_all_objects[j] = np.concatenate((normals_all_objects[j],normals[mask == j]), axis = 0)
           
        # whether to use a shape-completed mesh
        completed_shape_fn = self.configs.get('shape_completion_fn', None)
        pcds_all_objects_numpy = None
        pcd_mask = []
        self.visible_pcds = None
        self.visible_pcd_mask = []
        for j in range(self.get_num_objects()):
            print(j, self.get_num_objects())
            if j == self.robotIndex:
                continue
            background_idx = self.configs.get('background_idx', 0)
            obj_idx = self.configs.get('rigid_object_idx', 1)
            if j == obj_idx and completed_shape_fn is not None:
                mesh = o3d.io.read_triangle_mesh(self.configs['shape_completion_fn']) 
                if not mesh.has_triangles():
                    print("Error: The mesh does not have triangle information.")
                    exit()
                
                # Calculate normals
                mesh.compute_vertex_normals()
                # Get the normals as a numpy array
                normals = np.asarray(mesh.vertex_normals)
                points = np.asarray(mesh.vertices)

                # TripoSR mesh is BR colors are swapped
                colors = np.asarray(mesh.vertex_colors)
                colors[:, [0, 2]] = colors[:, [2, 0]]

                # fit the generated mesh to the point cloud
                # scale the observed_pcd roughly
                scale_obs = np.max(np.max(np.array(xyzs_all_objects[j]), axis = 0) - \
                    np.min(np.array(xyzs_all_objects[j]), axis = 0))
                scale_completion= np.max(np.max(points, axis = 0) - \
                    np.min(points, axis = 0))
                
                o3d_obs = o3d.geometry.PointCloud()
                o3d_obs.points = o3d.utility.Vector3dVector(np.array(xyzs_all_objects[j]))
                o3d_obs.normals = o3d.utility.Vector3dVector(np.array(normals_all_objects[j]))
                o3d_completion = o3d.geometry.PointCloud()
                o3d_completion.points = o3d.utility.Vector3dVector(points*scale_obs/scale_completion)
                o3d_completion.normals = o3d.utility.Vector3dVector(normals)
                o3d_completion.colors = o3d.utility.Vector3dVector(colors)

                T, s = register_pcd(o3d_completion, o3d_obs)
                o3d_completion.scale(s, center=o3d_completion.get_center())
                o3d_completion.transform(T)

                # downsample the object 
                print(f'Before downsampling, object has {len(np.asarray(o3d_completion.points))} points')
                o3d_completion = o3d_completion.voxel_down_sample(self.configs['initial_geometry_voxel_downasample_size'])
                print(f'After downsampling, object has {len(np.asarray(o3d_completion.points))} points')

                try:
                    o3d.visualization.draw_geometries([o3d_completion, o3d_obs], window_name = f'Final registration result for object {j}')
                except Exception as e:
                    print(f"An error occurred during open3d visualization: {e}")

                pcds = np.concatenate((np.asarray(o3d_completion.points), 
                                       np.asarray(o3d_completion.colors), 
                                       np.asarray(o3d_completion.normals)), axis = 1)
              
                n_pts = len(pcds)
                if pcds_all_objects_numpy is None:
                    pcds_all_objects_numpy = pcds
                else:
                    pcds_all_objects_numpy = np.concatenate((pcds_all_objects_numpy, pcds))
                pcd_mask += [j]*n_pts
            elif j == background_idx:
                pcds = np.concatenate((np.array(xyzs_all_objects[j]),np.array(rgbs_all_objects[j]), \
                                    normals_all_objects[j]), axis = 1)
                # fill the occluded region on the table 
                N = 500000
                xy = np.random.rand(N, 2)*np.array([0.4, 0.4]) - np.array([0.2, 0.2])
                zrgbn = np.zeros((N,7))
                zrgbn[:,-1] = 1         
                xyzrgbn = np.append(xy, zrgbn, axis = 1)
            
                # assign the color bases on neighbors
                tree = KDTree(pcds[:,0:3])
                distances, indices = tree.query(xyzrgbn[:,0:3], k=1)
                xyzrgbn[:,3:6] = pcds[indices,3:6]
                
                pcds= np.concatenate((pcds, xyzrgbn), axis = 0)
                print(f'Before background downsampling, {len(pcds)} points.')
                # downsample
                o3d_tmp = o3d.geometry.PointCloud()
                o3d_tmp.points = o3d.utility.Vector3dVector(pcds[:,0:3])
                o3d_tmp.colors = o3d.utility.Vector3dVector(pcds[:,3:6])
                o3d_tmp.normals = o3d.utility.Vector3dVector(pcds[:,6:9])
                o3d_tmp = o3d_tmp.voxel_down_sample(0.0025) 
                pcds = np.concatenate((np.asarray(o3d_tmp.points),np.asarray(o3d_tmp.colors), \
                                    np.asarray(o3d_tmp.normals)), axis = 1)
                print(f'After background downsampling, {len(pcds)} points.')
                if pcds_all_objects_numpy is None:
                    pcds_all_objects_numpy = pcds
                else:
                    pcds_all_objects_numpy = np.concatenate((pcds_all_objects_numpy, pcds))
                pcd_mask += [j]*len(pcds)
            else:
                pcds = np.concatenate((np.array(xyzs_all_objects[j]),np.array(rgbs_all_objects[j]), \
                                    normals_all_objects[j]), axis = 1)
                # downsample
                o3d_tmp = o3d.geometry.PointCloud()
                o3d_tmp.points = o3d.utility.Vector3dVector(pcds[:,0:3])
                o3d_tmp.colors = o3d.utility.Vector3dVector(pcds[:,3:6])
                o3d_tmp.normals = o3d.utility.Vector3dVector(pcds[:,6:9])
                o3d_tmp = o3d_tmp.voxel_down_sample(0.0025)
                pcds = np.concatenate((np.asarray(o3d_tmp.points),np.asarray(o3d_tmp.colors), \
                                    np.asarray(o3d_tmp.normals)), axis = 1)
                
                if pcds_all_objects_numpy is None:
                    pcds_all_objects_numpy = pcds[indices,:]
                else:
                    pcds_all_objects_numpy = np.concatenate((pcds_all_objects_numpy, pcds))

                
                pcd_mask += [j]*len(pcds)

            # calculate visible pcd
            pcds = np.concatenate((np.array(xyzs_all_objects[j]),np.array(rgbs_all_objects[j]), \
                                    normals_all_objects[j]), axis = 1)
            downsample_rate = self.configs.get('pcd_downsample_rate_stage_1', 1/20)
            n_pts = int(len(pcds)*downsample_rate)
            indices = np.random.choice(len(pcds), n_pts, replace = False)
            if self.visible_pcds is None:
                self.visible_pcds  = pcds[indices,:]
            else:
                self.visible_pcds  = np.concatenate((self.visible_pcds , pcds[indices,:]))
            self.visible_pcd_mask += [j]*n_pts

        print('Pointcloud processing done.. saving')
        self.pcds = np.array(pcds_all_objects_numpy)
        self.pcd_mask = np.array(pcd_mask)
        self.visible_pcds = np.array(self.visible_pcds)
        self.visible_pcd_mask = np.array(self.visible_pcd_mask)
      
        # visualize for debugging
        disp_pcd = o3d.geometry.PointCloud()
        disp_pcd.points = o3d.utility.Vector3dVector(self.pcds[:,:3]) #self.pcd_mask==1
        disp_pcd.colors = o3d.utility.Vector3dVector(self.pcds[:,3:6])
        disp_pcd.normals = o3d.utility.Vector3dVector(self.pcds[:,6:])
        try:
            o3d.visualization.draw_geometries([disp_pcd], window_name = 'Final processed pcd scene')
        except Exception as e:
            print(f"An error occurred: {e}")

        np.save(pcd_fn, self.pcds, allow_pickle=True)
        np.save(pcd_mask_fn, self.pcd_mask, allow_pickle=True)
        np.save(visible_pcd_fn, self.visible_pcds, allow_pickle=True)
        np.save(visible_pcd_mask_fn, self.visible_pcd_mask, allow_pickle=True)
        return

class DataloaderStage2(dataloader):
    def __init__(self, folder, zarr_name, configs, compute_init = True) -> None:
        super().__init__(os.path.join(folder, zarr_name), configs)
        self.folder = folder
        self.pcds = self.num_images*[0]
        self.pcd_mask = self.num_images*[0]
        self.name = zarr_name.split('.')[0]
        if compute_init:
            self._compute_initial_pcd()
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        return copy.copy(self.root['rgb_images'][i,:])/255.,\
            copy.copy(self.root['depth_images'][i,:])*self.depth_scale,\
            copy.copy(self.root['masks'][i,:]),\
            copy.copy(self.root['poses'][i,:]),\
            copy.copy(self.pcds[i]),copy.copy(self.pcd_mask[i])
    
    def get_item_no_mask(self, i):
        return copy.copy(self.root['rgb_images'][i,:])/255.,\
            copy.copy(self.root['depth_images'][i,:])*self.depth_scale,\
            
    def get_all_data(self):
        return copy.copy(self.root['rgb_images'][:])/255.,\
            copy.copy(self.root['depth_images'][:])*self.depth_scale,\
            copy.copy(self.root['masks'][:]),\
            copy.copy(self.root['poses'][:]),\
            copy.copy(self.pcds),copy.copy(self.pcd_mask)

    def get_pcd(self):
        return copy.copy(self.pcds), copy.copy(self.pcd_mask)
    
    def get_forces(self):
        return copy.copy(self.root['applied_forces'][:])
    
    def _compute_initial_pcd(self):
        pcd_fn = os.path.join(self.folder,  self.name + 'stage2_pcds.npy')
        pcd_mask_fn = os.path.join(self.folder,  self.name + 'stage2_pcd_masks.npy')
        if os.path.exists(pcd_fn):
            self.pcds = np.load(pcd_fn)
            self.pcd_mask = np.load(pcd_mask_fn)
            return 
        pcd_fn = os.path.join(self.folder, self.name + 'stage2_pcds.npz')
        pcd_mask_fn = os.path.join(self.folder, self.name + 'stage2_pcd_masks.npz')

        if os.path.exists(pcd_fn):
            def load_array_list(filename):
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"File {filename} not found")
                loaded = np.load(filename)
                array_list = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
                return array_list
            self.pcds = load_array_list(pcd_fn)
            self.pcd_mask = load_array_list(pcd_mask_fn)
            return 

        pcds_all_timesteps = []
        masks_all_timesteps = []
        for i in range(self.__len__()):
            print(i/self.__len__())
            rgb, depth, mask, pose, _,_ = self.__getitem__(i)
            mask = mask.reshape(-1)
            view_matrix = pose
            camera_T = se3.inv((list(view_matrix[0:3]) + list(view_matrix[4:7]) + list(view_matrix[8:11]), \
                        list(view_matrix[12:15])))
            camera_t = camera_T[1]
            xyzs = compute_pcd(rgb, depth, pose, self.get_projection_matrix())
            rgbs = rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]))

            
            # use voxel downsample to count how many points to subsample
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(np.array(xyzs))
            o3d_pcd.colors = o3d.utility.Vector3dVector(np.array(rgbs))
            voxel_size = self.configs.get('initial_geometry_voxel_downasample_size', 0.0025)
            downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size)
            # now random downsample
            n_pts = len(downsampled_pcd.points)
            indices = np.random.choice(len(xyzs), n_pts, replace = False)
            xyzs, rgbs, mask = xyzs[indices], rgbs[indices], mask[indices]

            normals = np.zeros((len(xyzs), 3))
            pcds = np.concatenate((xyzs, rgbs, normals), axis = 1)
            pcd_mask = np.array(mask)
            pcds_all_timesteps.append(pcds)
            masks_all_timesteps.append(pcd_mask)
        
        self.pcds = pcds_all_timesteps
        self.pcd_mask = masks_all_timesteps
        def save_array_list(array_list, filename):
            array_dict = {f'arr_{i}': arr for i, arr in enumerate(array_list)}
            np.savez(filename, **array_dict)
        save_array_list(self.pcds, pcd_fn)
        save_array_list(self.pcd_mask, pcd_mask_fn)
        return

    def get_N_time_steps(self):
        return copy.copy(self.root.attrs['numImages'])
    
    def get_time_stamps(self):
        return copy.copy(self.root['timestamps'][:])

    def get_robot_poses_vels(self):
        return copy.copy(self.root['robot_poses'][:]),\
            copy.copy(self.root['robot_linear_vels'][:])
    
    def get_obj_poses(self):
        return copy.copy(self.root['obj_poses'][:])
