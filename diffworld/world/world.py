import os
import copy

import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes as pytorch3dMesh
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import RasterizationSettings,MeshRenderer,MeshRasterizer, PointLights, TexturesVertex,SoftPhongShader
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, so3_exponential_map, quaternion_multiply, \
    quaternion_apply
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
from pytorch3d.io import IO
import ode
from icecream import ic
import open3d as o3d
from tqdm import tqdm

from diffworld.diffsdfsim.sdf_physics.physics3d.world import World3D
from diffworld.diffsdfsim.sdf_physics.physics3d.utils import get_tensor
from diffworld.diffsdfsim.sdf_physics.physics3d.constraints import TotalConstraint3D, LinearVelConstraint, ZeroVelConstraint
from diffworld.diffsdfsim.sdf_physics.physics3d.forces import Gravity3D, ExternalForce3D
from diffworld.utils import get_learning_rate_schedules,adjust_learning_rate, get_discrete_sphere, mse
from diffworld.shape_as_points.src.dpsr import DPSR
from diffworld.shape_as_points.src.model import PSR2Mesh
from diffworld.shape_as_points.src.utils import mc_from_psr


class BaseWorld:
    def __init__(self, configs, entities) -> None:
        return 

    def set_train(self):
        raise NotImplementedError("Subclasses should implement this method")

    def set_val(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _reset_objects_constraints(self):
        raise NotImplementedError("Subclasses should implement this method")

    def reset(self):
        self._reset_objects_constraints()
        del self.simulator
        self.simulator = World3D(self.objects,self.constraints, strict_no_penetration=False, \
            configs = self.configs, dt = self.configs['dt'], contact_callback=self.configs['contact_callback'],\
            eps=self.configs['contact_eps'], tol = self.configs['contact_eps'],
            time_of_contact_diff = self.configs['time_of_contact_diff'])
        return

    def get_t(self):
        return self.simulator.t


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        # Create a binary mask from the depth buffer
        mask = (fragments.zbuf[:, :, :, 0] > 0).float()
        return images, fragments.zbuf, mask

class SaPWorld(BaseWorld):
    def __init__(self, configs, entities, device = 'cuda') -> None:
        super().__init__(configs, entities)
        self.entities = entities
        self.objects = []
        self.robot_sim_id = None
        for idx, entity in enumerate(entities):
            self.objects.append(SaPObject(entity, configs))
            if entity[1] == 'robot':
                self.robot_sim_id = idx
        self.configs = configs
        self.device = device
        self.l1_loss = torch.nn.L1Loss(reduction="sum")

        self.raster_settings = RasterizationSettings(
            image_size=(self.configs['image_height'], self.configs['image_width']), 
            blur_radius=0, 
            faces_per_pixel=1, 
            bin_size = 0,
            # max_faces_per_bin = 100000,
        )
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]],
                   ambient_color=[[0.2, 0.2, 0.2]], diffuse_color=[[0.8, 0.8, 0.8]], specular_color=[[0., 0., 0.]])
        return

    def initialize(self, pcds, pcdMask):
        for obj in self.objects:
            obj.initialize_geometry(pcds[pcdMask==obj.ID, :])
        self._initialize_simulator()

    def _initialize_simulator(self):
        self.constraints = self.get_default_constraints()
        self.simulator = World3D(self.objects,self.constraints, strict_no_penetration=False, \
            configs = self.configs, dt = self.configs['dt'], contact_callback=self.configs['contact_callback'],\
            eps=self.configs['contact_eps'], tol = self.configs['contact_eps'],time_of_contact_diff = self.configs['time_of_contact_diff'])
        return   
    
    def initialize_from_stage_1(self, folder, robot_state, reset_mass_fric = False, reset_robot_fric=False):
        for obj in self.objects:
            if obj.type != 'robot':
                if obj.ID in self.configs['optimizable_phy_ids']:
                    load_fric = False
                else:
                    load_fric = True
                obj.load(os.path.join(folder,f'obj_{obj.ID}_best.pth'), load_fric)
                if reset_mass_fric:
                    mass = self.configs['init_masses'][obj.ID]
                    obj.mass = get_tensor(mass).to(self.device)
                    fric = self.configs['init_frictions'][obj.ID]
                    obj.fric_coeff = torch.tensor(fric).to(self.device)
                obj.set_optimizable_geom_parameters()
                obj.set_physical_parameters()
                obj.set_geom()
            else:
                pos, ori, v = robot_state
                obj.initialize_robot(pos, ori, v)
                if reset_robot_fric:
                    fric = self.configs['init_frictions'][obj.ID]
                    obj.fric_coeff = torch.tensor(fric).to(self.device)
        self._initialize_simulator()
    
    def load_stage_1(self, folder, reset_mass_fric = False):
        "Essentially initialize_from_stage_1() but without the robot"
        for obj in self.objects:
            if obj.ID in self.configs['optimizable_phy_ids']:
                load_fric = False
            else:
                load_fric = True
            obj.load(os.path.join(folder,f'obj_{obj.ID}_full.pth'), load_fric)
            if reset_mass_fric:
                mass = self.configs['init_masses'][obj.ID]
                obj.mass = get_tensor(mass).to(self.device)
                fric = self.configs['init_frictions'][obj.ID]
                obj.fric_coeff = torch.tensor(fric).to(self.device)
            obj.set_optimizable_geom_parameters()
            obj.set_physical_parameters()
            obj.set_geom()
        self._initialize_simulator()


    def initialize_robot_state(self, pos, ori, v):
        for b in self.objects:
            if b.type == 'robot':
                b.initialize_robot_state(pos, ori, v)

    def update_learning_rate(self, epoch, inertial_param_start_epoch = 0):
        for obj in self.objects:
            obj.adjust_learning_rate(epoch, inertial_param_start_epoch)

    def set_state(self, states):
        return

    def set_optimizable_com(self):
        for b in self.objects:
            if b.type != 'robot':
                b.set_optimizable_com()

    def simulate_step(self, post_step_update = False, first_epoch = False):
        had_contacts, fc = self.simulator.step(fixed_dt = True, time_halving = not first_epoch)
        q = None
        dq = None
        for obj in self.objects:
            if obj.type in ['obj', 'robot']:
                if q is None:
                    q = obj.p
                    dq = obj.v
                else:
                    q = torch.cat((q, obj.p))
                    dq = torch.cat((dq, obj.v))
        
        if post_step_update:
            for obj in self.objects:
                obj.pos = obj.p[4:]
                obj.rot = obj.p[0:4]

        #first check if there is robot
        if self.robot_sim_id is None:
            return q, dq, None
        else:
            return q, dq, fc
               
    def robot_has_contact(self):
        for idx, c in enumerate(self.simulator.start_contacts):
            a, b1, b2 = c
            if b1 == self.robot_sim_id or b2 == self.robot_sim_id:
                return True
        return False
    
    def undo_step(self):
        self.simulator.undo_step()
    
    def render(self, cam, background, device = 'cuda', mc_stepsize = None, render_robot = False):
        if self.configs['color_representation'] == 'point_color':
            for obj in self.objects:
                if obj.type == 'terrain':
                    pts, cs = obj.get_world_pointscolors(pass_color_grad = True)
            points = pts.float()
            colors = cs.float()
           
            point_cloud = Pointclouds(points=[points], features=[colors])
            # now render the background
            bg_raster_settings = PointsRasterizationSettings(
                image_size=(self.configs['image_height'], self.configs['image_width']), 
                radius = self.configs['pytorch3d_point_size'], 
                points_per_pixel = 8
            )
            bg_rasterizer = PointsRasterizer(cameras=cam, raster_settings=bg_raster_settings)
            bg_renderer = PointsRenderer(
                rasterizer=bg_rasterizer,
                compositor=NormWeightedCompositor() #
            )
            rgb = bg_renderer(point_cloud).squeeze(0)
            bg_mask = ~(rgb.abs() < 0.001).all(dim=2)

            # Extract zbuf (depth) from rasterized points
            rasterized_points = bg_rasterizer(point_cloud)
            zbuf = rasterized_points.zbuf

            # Process depth image
            depth = zbuf[..., 0]  
            depth = depth.squeeze()  
            
            if self.configs['pytorch3d_color_renderer'] == 'point':
                ##black background
                def get_object_mask(image_tensor, threshold=1e-6):
                    mask = (image_tensor.abs() > threshold).all(dim=2)
                    return mask.float()
                def paste_object(image_A, image_B, threshold=1e-6):
                    assert image_A.shape == image_B.shape, "Images must have the same shape"
                    mask = get_object_mask(image_B, threshold)
                    mask = mask.unsqueeze(-1).expand_as(image_A)
                    result = image_A * (1 - mask) + image_B * mask
                    return result, mask
                for obj in self.objects:
                    if obj.type == 'obj':
                        pts, cs = obj.get_world_pointscolors(pass_color_grad = True)
                points = pts.float()
                colors = cs.float()
                point_cloud = Pointclouds(points=[points], features=[colors])
                obj_rgb = bg_renderer(point_cloud).squeeze(0)
                rgb, mask = paste_object(rgb, obj_rgb)
            elif self.configs['pytorch3d_color_renderer'] == 'mesh':
                def get_object_mask(image_tensor, threshold=1 - 1e-6):
                    mask = (image_tensor.abs() < threshold).all(dim=2)
                    return mask.float()
                def paste_object(image_A, image_B, mask):
                    assert image_A.shape == image_B.shape, "Images must have the same shape"
                    mask = mask.unsqueeze(-1).expand_as(image_A)
                    result = image_A * (1 - mask) + image_B * mask
                    return result
                
                def paste_object_depth(image_A, image_B, mask):
                    assert image_A.shape == image_B.shape, "Images must have the same shape"
                    result = image_A * (1 - mask) + image_B * mask
                    return result

                for obj in self.objects:
                    if obj.type == 'obj':
                        mesh = obj.get_mesh(mc_stepsize)
                        break
               
                # use color grid
                v_normalized = mesh[3]*2 - 1 #[-1 ,1] #1 x N x 3
                # swap axes for grid_sample convention.
                v_normalized[:,:,[-1,-3]] = v_normalized[:,:,[-3,-1]]
                verts_rgb = F.grid_sample(obj.color_grid, v_normalized.unsqueeze(0).unsqueeze(0), 
                                    mode='bilinear', align_corners=True).squeeze(2).squeeze(2) #BxN
                verts_rgb = torch.transpose(verts_rgb, 1, 2)
                textures = TexturesVertex(verts_features=verts_rgb.to(device))
                mesh = pytorch3dMesh(verts = mesh[0].float(), faces = mesh[1].float(), \
                    verts_normals = mesh[2].float(), textures = textures)
                
                # pytorch3d for rendering 
                renderer = MeshRendererWithDepth(
                    rasterizer=MeshRasterizer(
                        cameras=cam, 
                        raster_settings=self.raster_settings
                    ),
                    shader=SoftPhongShader(
                        device=device, 
                        cameras=cam,
                        lights=self.lights
                    )
                    )
                obj_rgb, obj_depth, mask = renderer(mesh) #first dimension is the batch NxHxW 

                obj_rgb = obj_rgb.squeeze(0)[:,:,:3]
                mask = mask.squeeze(0)
                obj_depth = obj_depth.squeeze(0).squeeze(-1)
                rgb = paste_object(rgb, obj_rgb, mask)
                depth = paste_object_depth(depth, obj_depth, mask)
                if render_robot:
                    robot_points, robot_normals = self.objects[2].get_pointsnormals()
                    robot_colors = torch.tensor([0.7,0.1,0.1]).repeat(len(robot_points), 1).to(self.device)   
                    
                    o3d_pcd = o3d.geometry.PointCloud()
                    o3d_pcd.points = o3d.utility.Vector3dVector(robot_points.cpu().detach().numpy())
                    o3d_pcd.colors = o3d.utility.Vector3dVector(robot_colors.cpu().detach().numpy())
                    o3d_pcd.normals =  o3d.utility.Vector3dVector(robot_normals.cpu().detach().numpy())
                    
                    radii = [0.005, 0.01, 0.02, 0.04]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                            o3d_pcd, o3d.utility.DoubleVector(radii)
                        )
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_duplicated_vertices()
                    mesh.remove_non_manifold_edges()
                    textures = TexturesVertex(verts_features=torch.from_numpy(np.asarray(mesh.vertex_colors)).to(self.device).float().unsqueeze(0))
                    mesh = pytorch3dMesh(verts = torch.from_numpy(np.asarray(mesh.vertices)).to(self.device).float().unsqueeze(0), 
                        faces = torch.from_numpy(np.asarray(mesh.triangles)).to(self.device).float().unsqueeze(0), \
                        verts_normals = torch.from_numpy(np.asarray(mesh.vertex_normals)).to(self.device).float().unsqueeze(0),
                        textures = textures)
                    robot_rgb, robot_depth, robot_mask = renderer(mesh)
                    robot_rgb = robot_rgb.squeeze(0)[:,:,:3]
                    robot_mask = robot_mask.squeeze(0)
                    robot_depth = robot_depth.squeeze(0).squeeze(-1)
                    rgb = paste_object(rgb, robot_rgb, robot_mask)
                    depth = paste_object_depth(depth, robot_depth, robot_mask)
            else:
                raise Exception
        
            return rgb, depth, mask, bg_mask
    
    def zero_grad(self):
        for obj in self.objects:
            if hasattr(obj, 'optimizer'):
                obj.optimizer.zero_grad()
            if hasattr(obj, 'phy_optimizer'):
                obj.phy_optimizer.zero_grad()
            if hasattr(obj, 'com_optimizer'):
                obj.com_optimizer.step()
            if hasattr(obj, 'color_optimizer'):
                obj.color_optimizer.step()
            if hasattr(obj, 'scale_optimizer'):
                obj.scale_optimizer.step()
        if hasattr(self, 'color_optimizer'):
            self.color_optimizer.step()

    def step(self, optimize_sap = True, optimize_inertial = True):
        for obj in self.objects:
            if hasattr(obj, 'optimizer') and optimize_sap:
                obj.optimizer.step()
            if hasattr(obj, 'phy_optimizer'):
                obj.phy_optimizer.step()
            if hasattr(obj, 'com_optimizer') and optimize_inertial:
                obj.com_optimizer.step()
            if hasattr(obj, 'color_optimizer'):
                obj.color_optimizer.step()
            if hasattr(obj, 'scale_optimizer') and optimize_inertial:
                obj.scale_optimizer.step()
            obj.post_step_update()
            obj.v_sap_frame = None
        if hasattr(self, 'color_optimizer'):
            self.color_optimizer.step()
    
    def set_train(self):
        return
    
    def set_eval(self):
        return 
        
    def _reset_objects_constraints(self):
        for i, b in enumerate(self.objects):
            self.simulator.space.remove(b.geom)
            del b.geom
        del self.simulator.space
        if not os.path.exists(self.configs['temp_folder']):
            os.makedirs(self.configs['temp_folder'])
        for i,b in enumerate(self.objects):
            b.save(os.path.join(self.configs['temp_folder'], f'temp_obj_state_{b.ID}.pth'))

        self.objects = []
        for entity in self.entities:
            b = SaPObject(entity, self.configs)
            b.load(os.path.join(self.configs['temp_folder'], f'temp_obj_state_{b.ID}.pth'))
            self.objects.append(b)
            b.set_optimizable_geom_parameters()
            b.set_physical_parameters()
            b.set_geom()

        self.constraints = self.get_default_constraints()

    def _reset_objects_constraints_from_dict(self, folder, prefix = ''):
        for i, b in enumerate(self.objects):
            self.simulator.space.remove(b.geom)
            del b.geom
        del self.simulator.space

        self.objects = []
        for entity in self.entities:
            b = SaPObject(entity, self.configs)
            if b.type == 'robot':
               b.load(os.path.join(folder, prefix + f'obj_{b.ID}_full.pth'))

            else:
                b.load(os.path.join(folder, prefix + f'obj_{b.ID}_full.pth'))
            self.objects.append(b)
            b.set_optimizable_geom_parameters()
            b.set_physical_parameters()
            b.set_geom()
        self.constraints = self.get_default_constraints()

    def reset_stage_2(self, folder, prefix = ''):
        self._reset_objects_constraints_from_dict(folder, prefix)
        del self.simulator
        self.simulator = World3D(self.objects,self.constraints, strict_no_penetration=False, \
            configs = self.configs, dt = self.configs['dt'], contact_callback=self.configs['contact_callback'],\
            eps=self.configs['contact_eps'], tol = self.configs['contact_eps'],\
                  time_of_contact_diff = self.configs['time_of_contact_diff'])
        return


    def get_default_constraints(self):
        constraints = []
        for obj in self.objects:
            if obj.type == 'terrain':
                constraints.append(TotalConstraint3D(obj))
            if obj.type == 'robot':
                constraints.append(ZeroVelConstraint(obj, [0,1,2,5]))
        return constraints

    def apply_force(self, obj_id, force_func):
        for obj in self.objects:
            if obj.ID == obj_id:
                obj.add_force(ExternalForce3D(force_func))

    def get_current_state_dict(self):
        objs = []
        for b in self.objects:
            objs.append(b.get_current_state_dict())
        return objs
    
    def set_state_from_dict(self, dict_list):
        for b,d in zip(self.objects, dict_list):
            b.set_state_from_dict(d)
        self.simulator.find_contacts()
        return
        
    def set_robot_vel(self, vel):
        self.constraints = self.get_default_constraints()
        for b in self.objects:
            if b.type == 'robot':
                self.constraints.append(LinearVelConstraint(b, vel))
                self.simulator.reset_eq_constraints(self.constraints)
                return
            
    def get_largest_penetrations(self):
        robot_obj_penetration_depths = []
        obj_terrain_penetration_depths = []
        for contact in self.simulator.contacts:
            types = [contact[1], contact[2]]
            if 2 in types:
                robot_obj_penetration_depths.append(contact[0][3].item())
            if 0 in types:
                obj_terrain_penetration_depths.append(contact[0][3].item())
        max_obj_terrain = 0
        max_robot_obj = 0
        if len(obj_terrain_penetration_depths) > 0:
            max_obj_terrain = np.max(obj_terrain_penetration_depths)
        if len(robot_obj_penetration_depths) > 0:
            max_robot_obj = np.max(robot_obj_penetration_depths)
        return max_obj_terrain, max_robot_obj

    def save_GS(self, path):
        torch.save(self.GS.capture(), path)

    def GS_postprocess(self):
        pcd = None
        for b in self.objects:
            if b.type != 'robot':
                p, c = b.get_world_pointscolors()
                if pcd is None:
                    pcd = p
                else:
                    pcd = torch.cat((pcd, p), axis = 0)
        self.GS.postprocess(pcd.float())


class Object:
    def __init__(self, entity, configs, device = 'cuda') -> None:
        self.ID = entity[0]
        self.type = entity[1]
        if self.type == 'robot':
            self.robot_geom = entity[2]
            self.robot_param = entity[3:]
        self.sdf = None
        self.device = device
        self.forces = []
        self.configs = configs

        self.eps = torch.tensor(math.fabs(configs['contact_eps'])).to(device)
        self._set_base_tensor(locals().values())
        mass = configs['init_masses'][self.ID]
        self.mass = get_tensor(mass).to(self.device)
        self.com = torch.from_numpy(np.array([0.,0.,0.])).to(device)
        self.inertia_scale = torch.from_numpy(np.array([1.,1.,1.])).to(device)
        fric = configs['init_frictions'][self.ID]
        self.fric_coeff = torch.tensor(fric).to(device)
        self.restitution = torch.tensor(0).to(device)

    def _custom_get_ang_inertia(self, mass, com, scale):
        diag_terms = [self.side_lengths[1]**2+self.side_lengths[2]**2,self.side_lengths[0]**2+self.side_lengths[2]**2,\
                      self.side_lengths[0]**2+self.side_lengths[1]**2]
        I = mass*torch.diag(scale)*torch.diag(torch.tensor(diag_terms)).to(self.device) / 12.0
        return I

    def initialize_geometry(self, pcd):
        return 
        
    def add_force(self, f):
        self.forces.append(f)
        f.set_body(self)

    def move(self, dt, update_geom_rotation=True):
        new_p = torch.cat([quaternion_multiply(matrix_to_quaternion(so3_exponential_map(self.v[:3].unsqueeze(0) * dt)),
                                               self.p[:4]).squeeze(),
                           self.p[4:] + self.v[3:] * dt])
        self.set_p(new_p, update_geom_rotation)

    
    def set_p(self, new_p, update_geom_rotation=False, update_ang_inertia=False):
        self.p = new_p
        # Reset memory pointers
        self.rot = self.p[0:4]
        self.pos = self.p[4:]
        self.R = quaternion_to_matrix(self.rot)

        self.geom.setPosition(self.pos)
        if update_geom_rotation:
            self.geom.setQuaternion(self.rot)

    def add_no_contact(self, other):
        self.geom.no_contact.add(other.geom)
        other.geom.no_contact.add(self.geom)


    def apply_forces(self, t):
        if len(self.forces) == 0:
            return self.v.new_zeros(len(self.v))
        else:
            return sum([f.force(t) for f in self.forces])

    def _set_base_tensor(self, args):
        """Check if any tensor provided and if so set as base tensor to
           use as base for other tensors' dtype, device and layout.
        """
        if hasattr(self, '_base_tensor') and self._base_tensor is not None:
            return

        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._base_tensor = arg
                return

        # if no tensor provided, use defaults
        self._base_tensor = get_tensor(0, base_tensor=None)
        return

class SaPObject(Object):
    def __init__(self, entity, configs, device = 'cuda') -> None:
        super().__init__(entity, configs, device) 
        self.lr_schedules = get_learning_rate_schedules(configs,'SaPLearningRateSchedule')
        self.phy_lr_schedules =  get_learning_rate_schedules(self.configs,'PhyLearningRateSchedule')
        self.color_lr_schedules = get_learning_rate_schedules(self.configs,'ColorLearningRateSchedule')
        self.color_terrain_lr_schedules = get_learning_rate_schedules(self.configs,'ColorTerrainLearningRateSchedule')
        self.com_lr_schedules =  get_learning_rate_schedules(self.configs,'ComLearningRateSchedule')
        self.scale_lr_schedules =  get_learning_rate_schedules(self.configs,'ScaleLearningRateSchedule')
        self.v_sap_frame = None
        self.f_sap_mesh = None
        self.n_sap_mesh = None
        self.v_sap_mesh = None
        # initialize DPSR
        if self.type == 'obj':
            self.dpsr = DPSR(res=(self.configs['SaP_grid_res'], 
                                self.configs['SaP_grid_res'], 
                                self.configs['SaP_grid_res']), 
                            sig=self.configs['SaP_psr_sigma'])
            self.dpsr = self.dpsr.to(device)
            self.psr2mesh = PSR2Mesh.apply

    def adjust_learning_rate(self, epoch, inertial_param_start_epoch = 0):
        if hasattr(self, 'sap_inputs') and hasattr(self, 'optimizer'):
            adjust_learning_rate(self.lr_schedules, self.optimizer, epoch)
        if hasattr(self, 'phy_optimizer'):
            adjust_learning_rate(self.phy_lr_schedules, self.phy_optimizer, epoch)
        if self.type == 'terrain':
            adjust_learning_rate(self.color_terrain_lr_schedules, self.color_optimizer, epoch)
        else:
            if hasattr(self, 'color_optimizer'):
                adjust_learning_rate(self.color_lr_schedules, self.color_optimizer, epoch)
        if hasattr(self, 'com_optimizer'):
            adjust_learning_rate(self.com_lr_schedules, self.com_optimizer, epoch - inertial_param_start_epoch)
        if hasattr(self, 'scale_optimizer'):
            adjust_learning_rate(self.scale_lr_schedules, self.scale_optimizer, epoch  - inertial_param_start_epoch)

    def initialize_geometry(self, pcd):
        if self.type == 'robot':
           raise Exception
        
        if self.type == 'obj' and self.configs.get('initial_sap_downsample', True):
            N = pcd.shape[0]
            indices = np.random.choice(N, N//3, replace=False)
            pcd = pcd[indices]

        normals = pcd[:,6:9]
        colors = pcd[:, 3:6]

        pcd = pcd[:, 0:3]
        pcd_world = copy.copy(pcd)
        if self.configs['reset_obj_pos']:
            self.t = np.zeros(3)
        else:
            self.t = np.mean(pcd, axis = 0)
        self.R = np.eye(3)
        pcd = pcd - self.t
        max_side = np.max(np.max(pcd, axis = 0) - np.min(pcd, axis = 0))
        self.scale = 2/max_side #inside a unit sphere
        self.side_lengths = torch.from_numpy(np.max(pcd, axis = 0) - np.min(pcd, axis = 0)).to(self.device)
        if self.side_lengths[2] < 0.005:
            self.side_lengths[2] = 0.005
        self.R = torch.from_numpy(self.R).to(self.device)
        self.scale_tensor = torch.tensor(self.scale).to(self.device)
        
        # These are the diffsdfsim properties for bodies
        self.rot = torch.tensor([1,0,0,0]).to(self.device) # quaterion
        self.pos = torch.from_numpy(self.t).to(self.device)
        self.p = torch.concatenate((self.rot, self.pos))
        self.v = torch.tensor([0.]*6).to(self.device).float()
        self.init_v = torch.tensor([0.]*6).to(self.device).float()
        self.colors = torch.from_numpy(colors).to(self.device)
      
        if self.type == 'obj':
            # SAP requires pts to be in the range of [0 ,1), make it in [0.15, 0.85]
            self.sap_scale = self.scale*0.7/2
            self.sap_points = pcd*self.sap_scale 
            self.sap_offset = np.min(self.sap_points, axis = 0) - 0.15
            self.sap_points -= self.sap_offset 
            self.sap_normals = normals
            # now sap points is in 0.15 to 0.85
            self.sap_points = torch.from_numpy(self.sap_points).unsqueeze(0).to(self.device)
            self.sap_points = torch.log(self.sap_points/(1 - self.sap_points)).float()
            self.sap_normals = torch.from_numpy(self.sap_normals).unsqueeze(0).to(self.device)
            self.sap_inputs = torch.cat([self.sap_points, self.sap_normals], axis = -1).float().to(self.device) # 1 X N X 6
            self.sap_offset = torch.from_numpy(self.sap_offset).to(self.device)
            self.sap_scale = torch.tensor(self.sap_scale).to(self.device)
            if self.configs['optimize_geometry']:
                self.sap_inputs.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    [
                        {
                            "params": [self.sap_inputs],
                            "lr": self.lr_schedules[0].get_learning_rate(0),
                        },
                    ])
            if self.configs['optimize_color'] and self.configs['color_representation'] == 'point_color':
                # 1 batch, 1 channel, D, H, W, [-1, 1] range
                if self.configs['pytorch3d_color_renderer'] == 'mesh':
                    arbitrary_color_grid = self.configs.get('arbitrary_color_grid', False)
                    if arbitrary_color_grid: # start with arbitray green color
                        color = torch.tensor([0.1,0.7,0.1]).to(self.device)
                        N = self.configs['SaP_grid_res']+1
                        self.color_grid = color.repeat(N, N, N, 1).permute(3, 0, 1, 2).unsqueeze(0)
                        self.color_grid.requires_grad = True
                        self.color_optimizer = torch.optim.Adam(
                        [
                                {
                                    "params": [self.color_grid],
                                    "lr": self.color_lr_schedules[0].get_learning_rate(0),
                                },
                            ])
                    else: # start with the colors of the initial geometry guess
                        self.color_grid = torch.ones(1,3,self.configs['SaP_grid_res']+1, \
                                                    self.configs['SaP_grid_res']+1,\
                                                    self.configs['SaP_grid_res']+1).to(self.device)*-1.
                        self.color_grid.requires_grad = True
                        self.color_optimizer = torch.optim.Adam(
                        [
                                {
                                    "params": [self.color_grid],
                                    "lr": self.color_lr_schedules[0].get_learning_rate(0),
                                },
                            ])

                        # initialize the color grid from colors (this only affects grids that contain the mesh vertices)
                        def tune_grid(multiplier):
                            pts = (torch.sigmoid(self.sap_inputs[:,:,:3])*2 - 1).clone().detach()*multiplier
                            pts.requires_grad = False
                            for epoch in tqdm(range(self.configs['finetune_color_grid_epoch']), desc="Color grid initialization"):
                                self.color_optimizer.zero_grad()
                                pred_rgb = F.grid_sample(self.color_grid, pts.unsqueeze(0).unsqueeze(0), 
                                            mode='bilinear', align_corners=True).squeeze(2).squeeze(2) #BxN
                                pred_rgb = torch.transpose(pred_rgb, 1, 2)
                                loss = mse(pred_rgb.float(), self.colors.float())
                                loss.backward()
                                self.color_optimizer.step()
                        tune_grid(1.)
                       
                        # grid color assigning to nearby grid nodes, which are not tuned from tune_grid
                        from scipy.spatial import cKDTree
                        def assign_nearest_colors(grid, invalid_value=-1., chunk_size=100000):
                            device = grid.device
                            grid_shape = grid.shape
                            grid = grid.squeeze(0).permute(1, 2, 3, 0)  # Change to (N, N, N, 3)
                            
                            # Find valid colors
                            valid_mask = ~torch.all(torch.isclose(grid, torch.tensor([invalid_value], device=device)), dim=-1)
                            valid_positions = torch.nonzero(valid_mask).float().cpu().numpy()
                            valid_colors = grid[valid_mask].cpu().numpy()
                            if len(valid_colors) == 0:
                                raise ValueError("No valid colors found in the grid.")
                            
                            tree = cKDTree(valid_positions)
                            result = torch.zeros_like(grid)
                            total_points = grid_shape[2] * grid_shape[3] * grid_shape[4]
                            for start in range(0, total_points, chunk_size):
                                end = min(start + chunk_size, total_points)
                                x, y, z = np.unravel_index(np.arange(start, end), (grid_shape[2], grid_shape[3], grid_shape[4]))
                                chunk_positions = np.column_stack([z, y, x]).astype(np.float32)
                                _, nearest_indices = tree.query(chunk_positions, k=1)
                                chunk_colors = torch.tensor(valid_colors[nearest_indices], device=device)
                                result.view(-1, 3)[start:end] = chunk_colors
                            
                            result[valid_mask] = grid[valid_mask]
                            result = result.permute(3, 0, 1, 2).unsqueeze(0)
                            
                            return result

                        self.color_grid = assign_nearest_colors(self.color_grid.clone().detach())
                        self.color_grid.requires_grad = True
                        self.color_optimizer = torch.optim.Adam(
                        [
                                {
                                    "params": [self.color_grid],
                                    "lr": self.color_lr_schedules[0].get_learning_rate(0),
                                },
                            ])
                elif self.configs['pytorch3d_color_renderer'] == 'point':
                    self.colors.requires_grad = True
                    self.color_optimizer = torch.optim.Adam(
                    [
                            {
                                "params": [self.colors],
                                "lr": self.color_lr_schedules[0].get_learning_rate(0),
                            },
                        ])

            self.update_psr()
        elif self.type == 'terrain':
            self.points_world = torch.from_numpy(pcd_world).to(self.device)
            self.normals_world = torch.from_numpy(normals).to(self.device)
            if self.configs['optimize_color'] and self.configs['color_representation'] == 'point_color' \
                and self.configs['optimize_terrain_color']:
                self.colors.requires_grad = True
                self.color_optimizer = torch.optim.Adam(
                    [
                            {
                                "params": [self.colors],
                                "lr": self.color_terrain_lr_schedules[0].get_learning_rate(0),
                            },
                        ])
        else:
            raise NotImplementedError
        
        #setup for diffsdfsim
        self.set_geom()
        self.set_physical_parameters()
        return
    
    def initialize_robot(self, pos, ori, v):
        self.mass = torch.tensor(self.robot_param[1]).to(self.device)
        self.fric_coeff = torch.tensor(self.robot_param[2]).to(self.device)
        self.scale_tensor = torch.tensor(1./self.robot_param[0]).to(self.device)
        self.side_lengths =  torch.tensor([1./self.robot_param[0],1./self.robot_param[0],1./self.robot_param[0]]).to(self.device)
        self.com = torch.tensor([0.,0.,0.]).to(self.device)
        self.inertia_scale = torch.tensor([1.,1.,1.]).to(self.device)
        rot_inertia = self._custom_get_ang_inertia(self.mass, self.com,self.inertia_scale)
        self.M = torch.zeros((6,6)).to(self.device)
        self.M[:3,:3] = rot_inertia
        self.M[3:, 3:] = torch.eye(3).to(self.device)*self.mass
        self.M = self.M.float() 
        if self.robot_geom == 'sphere':
            self.robot_points = torch.from_numpy(get_discrete_sphere(r = self.robot_param[0])).to(self.device)
            self.geom = ode.GeomBox(None, (2/self.scale_tensor).expand(3) * 2 + 2 * self.eps.item())
            self.geom.no_contact = set()
        self.add_force(Gravity3D())
        self.set_state(pos, ori, v)
        

    def get_current_state_dict(self):
        save_dict = {}
        save_dict['R'] = self.R.to(self.device)
        save_dict['pos'] = self.pos
        save_dict['rot'] = self.rot
        save_dict['p'] = self.p
        save_dict['v'] = self.v
        save_dict['init_v'] = self.init_v
        return save_dict

    def set_state_from_dict(self, dict):
        self.R = dict['R'].to(self.device)
        self.pos = dict['pos'].to(self.device)
        self.rot = dict['rot'].to(self.device)
        self.p = torch.concatenate((self.rot, self.pos))
        self.v = dict['v'].to(self.device)
        self.init_v = dict['init_v'].to(self.device)

    def set_state(self, pos, ori, v):
        self.pos = torch.from_numpy(pos).to(self.device)
        self.rot = torch.tensor(ori).to(self.device) # quaterion
        self.R = quaternion_to_matrix(self.rot).to(self.device)
        self.p = torch.concatenate((self.rot, self.pos))
        self.v = torch.tensor(v).to(self.device).float()
        self.init_v = torch.tensor(v).to(self.device).float()
        self.geom.setPosition(self.pos)

    def update_psr(self):
        if self.type != 'obj':
            return None
        """ Obtain the PSR indicator grid, given the current oriented point cloud"""
        points, normals = self.sap_inputs[...,:3], self.sap_inputs[...,3:]
        if self.configs['SaP_apply_sigmoid']:
            points = torch.sigmoid(points)
        if self.configs['SaP_normal_normalize']:
            normals = normals / normals.norm(dim=-1, keepdim=True)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        self.psr_grid = torch.tanh(psr_grid)

        # normal_grid
        normal_grid = torch.zeros((self.configs['SaP_grid_res'],self.configs['SaP_grid_res'],self.configs['SaP_grid_res'],3))

        psr_grid_squeezed = self.psr_grid.squeeze(0).squeeze(0)
        # normal x
        psr_grid_shift = torch.zeros((self.configs['SaP_grid_res'],self.configs['SaP_grid_res'],\
                                    self.configs['SaP_grid_res'])).to(self.device)
        psr_grid_shift[1:,:,:] = psr_grid_squeezed[0:self.configs['SaP_grid_res']-1, :,:]

        normal_grid[1:, :, :, 0] = psr_grid_squeezed[1:, :, :] - psr_grid_shift[1:,:,:]
        normal_grid[0, :, :, 0] = normal_grid[1, :, :, 0]
        # normal y
        psr_grid_shift = torch.zeros((self.configs['SaP_grid_res'],self.configs['SaP_grid_res'],\
                                    self.configs['SaP_grid_res'])).to(self.device)
        psr_grid_shift[:,1:,:] = psr_grid_squeezed[:, 0:self.configs['SaP_grid_res']-1, :]
        normal_grid[:, 1:, :, 1] = psr_grid_squeezed[:, 1:, :] - psr_grid_shift[:,1:,:]
        normal_grid[:, 0, :, 1] = normal_grid[:, 1, :, 1]
        # normal z
        psr_grid_shift = torch.zeros((self.configs['SaP_grid_res'],self.configs['SaP_grid_res'],\
                                    self.configs['SaP_grid_res'])).to(self.device)
        psr_grid_shift[:,:,1:] = psr_grid_squeezed[:,:,0:self.configs['SaP_grid_res']-1]
        normal_grid[:, :, 1:, 2] = psr_grid_squeezed[:, :, 1:] - psr_grid_shift[:,:,1:]
        normal_grid[:, :, 0, 2] = normal_grid[:, :, 1, 2]
    
        # normalize 
        magnitudes = torch.norm(normal_grid, dim=-1, keepdim=True)
        normal_grid = normal_grid/magnitudes

        # Handle zero vectors (if any)
        zero_mask = magnitudes.squeeze(-1) == 0
        normal_grid[zero_mask] = torch.tensor([0.0, 0.0, 1.0])
        self.psr_normal = normal_grid.unsqueeze(0).unsqueeze(0).double().to(self.device)

    def get_mesh(self, mc_stepsize = None):
        if self.type != 'obj':
            return None
        
        if self.v_sap_frame is None: # this has been calculated
            # mc + to tensor
            if mc_stepsize is not None:
                self.v_sap_mesh, self.f_sap_mesh, self.n_sap_mesh = self.psr2mesh(self.psr_grid, mc_stepsize) # v is in the range of [0, 1)
            else:
                self.v_sap_mesh, self.f_sap_mesh, self.n_sap_mesh = self.psr2mesh(self.psr_grid, self.configs['mc_stepsize']) # v is in the range of [0, 1)

            # first scale back for sap, and turn it into obj frame (resulting: v in range -1 to 1)
            self.v_sap_frame = (self.v_sap_mesh + self.sap_offset)/self.sap_scale
            
        global_v = quaternion_apply(self.rot, self.v_sap_frame)
        global_v = global_v + self.pos
        return global_v, self.f_sap_mesh,self.n_sap_mesh, self.v_sap_mesh

    
    def get_world_sap_points(self):
        pts = torch.sigmoid(self.sap_inputs[0,:,:3])
        pts = (pts + self.sap_offset)/self.sap_scale
        pts = quaternion_apply(self.rot, pts) + self.pos     
        return pts

    def get_o3d_mesh(self):
        with torch.no_grad():
            v, f, _ = mc_from_psr(self.psr_grid,
                    zero_level=0, real_scale=True, step_size=self.configs['mc_stepsize'])
            v = (v + self.sap_offset.cpu().detach().numpy())/self.sap_scale.cpu().detach().numpy()
            v = v + self.pos.cpu().detach().numpy()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            return mesh
        
    def get_pointsnormals(self):
        if self.type == 'robot':
            row_norms = torch.norm(self.robot_points, p=2, dim=1, keepdim=True)
            normals = self.robot_points / row_norms
            return self.robot_points + self.pos, normals
        if self.type == 'terrain':
            if self.configs.get('set_world_normals', False):
                self.normals_world = torch.from_numpy(np.array([[0.,0.,1.] \
                    for i in range(len(self.points_world))])).to(self.device)
                return self.points_world, self.normals_world
        if not hasattr(self, 'normals_world') or \
            self.normals_world is None:
            self.normals_world = torch.from_numpy(np.array([[0.,0.,1.] \
                for i in range(len(self.points_world))])).to(self.device)
        return self.points_world, self.normals_world
        
    def post_step_update(self):
        if hasattr(self, 'sap_inputs'):
            self.update_psr()
        rot_inertia = self._custom_get_ang_inertia(self.mass, self.com,self.inertia_scale)
        self.M = torch.zeros((6,6)).to(self.device)
        self.M[:3,:3] = rot_inertia
        self.M[3:, 3:] = torch.eye(3).to(self.device)*self.mass
        self.M = self.M.float()

    def save(self, fn):
        save_dict = {}
        if hasattr(self, 'sap_inputs'):
            save_dict['sap_inputs'] = self.sap_inputs
            save_dict['sap_offset'] = self.sap_offset
            save_dict['sap_scale'] = self.sap_scale
        if hasattr(self, 'color_grid'):
            save_dict['color_grid'] = self.color_grid
        if hasattr(self, 'points_world'):
            save_dict['points_world'] = self.points_world
        if hasattr(self, 'normals_world'):
            save_dict['normals_world'] = self.normals_world
        if hasattr(self, 'robot_points'):
            save_dict['robot_points'] = self.robot_points
        if hasattr(self, 'colors'):
            save_dict['colors'] = self.colors
        save_dict['fric_coeff'] = self.fric_coeff
        save_dict['mass'] = self.mass
        save_dict['R'] = self.R.to(self.device)
        save_dict['scale_tensor'] = self.scale_tensor
        save_dict['pos'] = self.pos
        save_dict['rot'] = self.rot
        save_dict['p'] = self.p
        save_dict['v'] = self.v
        save_dict['init_v'] = self.init_v
        save_dict['com'] = self.com
        save_dict['inertia_scale'] = self.inertia_scale
        save_dict['side_lengths'] = self.side_lengths
        torch.save(save_dict, fn)

    def load(self, fn, load_optimizable_fric = True):
        loaded_dict = torch.load(fn)
        #print(self.ID, loaded_dict.keys())
        if 'sap_inputs' in loaded_dict.keys():
            self.sap_inputs = loaded_dict['sap_inputs'].to(self.device)
            if self.configs['optimize_geometry']:
                self.sap_inputs.requires_grad = True
            else:
                self.sap_inputs.requires_grad = False
            self.sap_offset = loaded_dict['sap_offset']
            self.sap_scale = loaded_dict['sap_scale']
        if 'color_grid' in loaded_dict.keys():
            self.color_grid = loaded_dict['color_grid'].to(self.device)
            if self.configs['optimize_color']:
                self.color_grid.requires_grad = True
            else:
                self.color_grid.requires_grad = False
        if 'points_world' in loaded_dict.keys():
            self.points_world = loaded_dict['points_world']
        if 'colors' in loaded_dict.keys():
            self.colors = loaded_dict['colors']
        if 'normals_world' in loaded_dict.keys():
            self.normals_world = loaded_dict['normals_world']
        if 'robot_points' in loaded_dict.keys():
            self.robot_points = loaded_dict['robot_points']
        if 'fric_coeff' in loaded_dict.keys():
            if load_optimizable_fric:
                self.fric_coeff = loaded_dict['fric_coeff']
        self.mass = loaded_dict['mass'].to(self.device)
        self.R = loaded_dict['R'].to(self.device)
        self.scale_tensor = loaded_dict['scale_tensor'].to(self.device)
        self.pos = loaded_dict['pos'].to(self.device)
        self.rot= loaded_dict['rot'].to(self.device)
        self.p = torch.concatenate((self.rot, self.pos))
        self.v = loaded_dict['init_v'].to(self.device)
        self.init_v = loaded_dict['init_v'].to(self.device)
        self.com = loaded_dict['com']
        self.inertia_scale = loaded_dict['inertia_scale']
        self.side_lengths = loaded_dict['side_lengths']

    def set_optimizable_geom_parameters(self):
        if self.type == 'obj':
            if self.configs['optimize_geometry']:
                self.sap_inputs.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    [
                        {
                            "params": [self.sap_inputs],
                            "lr": self.lr_schedules[0].get_learning_rate(0),
                        },
                    ])
            else:
                self.sap_inputs.requires_grad = False
            if self.configs['optimize_color'] and self.configs['color_representation'] == 'point_color':
                if self.configs['pytorch3d_color_renderer'] == 'mesh':
                    self.color_grid.requires_grad = True
                    self.color_optimizer = torch.optim.Adam(
                        [
                            {
                                "params": [self.color_grid],
                                "lr": self.color_lr_schedules[0].get_learning_rate(0),
                            },
                        ])
                elif self.configs['pytorch3d_color_renderer'] == 'point':
                    self.colors.requires_grad = True
                    self.color_optimizer = torch.optim.Adam(
                    [
                            {
                                "params": [self.colors],
                                "lr": self.color_lr_schedules[0].get_learning_rate(0),
                            },
                        ])
            self.update_psr()
        elif self.type == 'terrain':
            if self.configs['optimize_color'] and self.configs['color_representation'] == 'point_color'\
            and self.configs['optimize_terrain_color']:
                self.colors.requires_grad = True
                self.color_optimizer = torch.optim.Adam(
                    [
                            {
                                "params": [self.colors],
                                "lr": self.color_terrain_lr_schedules[0].get_learning_rate(0),
                            },
                        ])
        elif self.type == 'robot':
            pass
        else:
            raise NotImplementedError
        return


    def log_mesh(self, folder, fn):
        with torch.no_grad():
            # Save mesh
            v,f,n,v_normalized = self.get_mesh()
            v_normalized = v_normalized*2 - 1 #[-1 ,1] #1 x N x 3
            v_normalized[:,:,[-1,-3]] = v_normalized[:,:,[-3,-1]]
            verts_rgb = F.grid_sample(self.color_grid, v_normalized.unsqueeze(0).unsqueeze(0), 
                                mode='bilinear', align_corners=True).squeeze(2).squeeze(2) 
            verts_rgb = torch.clip(torch.transpose(verts_rgb, 1, 2).to(self.device),0,1)
            verts_bgr = verts_rgb.clone()
            verts_bgr[:,:,[0,2]] = verts_bgr[:,:,[2,0]]
            textures_rgb = TexturesVertex(verts_features=verts_rgb)
            textures_bgr = TexturesVertex(verts_features=verts_bgr)
            mesh_rgb = pytorch3dMesh(verts = v.float(), faces = f.float(), \
                        verts_normals = n, textures = textures_rgb)
            mesh_bgr = pytorch3dMesh(verts = v.float(), faces = f.float(), \
                        verts_normals = n, textures = textures_bgr)
            io = IO()
            io.save_mesh(mesh_bgr, os.path.join(folder, fn + '.ply'), binary= False, colors_as_uint8=True)
      

    def set_geom(self):
        self.geom = ode.GeomBox(None, (2/self.scale_tensor).expand(3) * 2 + 2 * self.eps.item())
        self.geom.setPosition(self.pos)
        self.geom.no_contact = set()

    def set_physical_parameters(self):
        if self.type in ['obj','robot']:
            self.add_force(Gravity3D())
        #set up phy parameters for optimization
        param_list = []
        for param in self.configs['optimizable_phy_param']:
            if self.ID in self.configs['optimizable_phy_ids']:
                if param == 'mass':
                    if self.type == 'obj':
                        self.mass.requires_grad = True
                        param_list.append(self.mass)
                if param == 'friction':
                    self.fric_coeff.requires_grad = True
                    param_list.append(self.fric_coeff)
        if self.ID in self.configs['optimizable_phy_ids'] and 'com' in self.configs['optimizable_phy_param']:
            self.com_optimizer = torch.optim.Adam(
                [
                    {
                        "params": [self.com],
                        "lr": self.com_lr_schedules[0].get_learning_rate(0),
                    },
                ])
            self.scale_optimizer = torch.optim.Adam(
                [
                    {
                        "params": [self.inertia_scale],
                        "lr": self.scale_lr_schedules[0].get_learning_rate(0),
                    },
                ])
            self.com.requires_grad = True
            self.inertia_scale.requires_grad = True
        rot_inertia = self._custom_get_ang_inertia(self.mass, self.com,self.inertia_scale)
        self.M = torch.zeros((6,6)).to(self.device)
        self.M[:3,:3] = rot_inertia
        self.M[3:, 3:] = torch.eye(3).to(self.device)*self.mass
        self.M = self.M.float() 
        if len(param_list) > 0:
            self.phy_optimizer = torch.optim.Adam(
                [
                    {
                        "params": param_list,
                        "lr": self.phy_lr_schedules[0].get_learning_rate(0),
                    },
                ])


    def get_world_pointscolors(self, pass_color_grad = False):
        if pass_color_grad:
            if self.type == 'robot':
                #TODO: handle orientation
                row_norms = torch.norm(self.robot_points, p=2, dim=1, keepdim=True)
                normals = self.robot_points / row_norms
                return self.robot_points.clone().detach() + self.pos, None
            elif self.type == 'terrain':
                return self.points_world.clone().detach(), self.colors
            elif self.type == 'obj':
                return self.get_world_sap_points().clone().detach(), self.colors
        else:
            if self.type == 'robot':
                row_norms = torch.norm(self.robot_points, p=2, dim=1, keepdim=True)
                normals = self.robot_points / row_norms
                return self.robot_points.clone().detach(), None
            elif self.type == 'terrain':
                return self.points_world.clone().detach(), self.colors.clone().detach()
            elif self.type == 'obj':
                return self.get_world_sap_points().clone().detach(), self.colors.clone().detach()