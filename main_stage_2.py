import os
import math, json

from pytorch3d.structures import Meshes as pytorch3dMesh
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
import torch 
import torchvision.transforms as transforms
from icecream import ic
import yaml
from tqdm import tqdm
import numpy as np
import cv2
from cprint import *
import wandb
import argparse
from klampt.math import so3

from diffworld.data import DataloaderStage2
from diffworld.world import SaPWorld
from diffworld.utils import convert_camera, convert_camera_pytorch3D,  l1_loss, mse, ssim,PartialChamferLoss,convert_camera_pytorch3D_from_camM

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Script for running stage 2 training")
parser.add_argument("--config",required=True,type=str,help="Path to the configuration file")
parser.add_argument('--use_wandb', action='store_true', default=False,
                    help='Use Weights & Biases for logging (default: False)')
args = parser.parse_args()
config_fn = args.config
use_wandb = args.use_wandb
set_seed(1)
device = 'cuda'
transform = transforms.ToTensor()

# Load config
with open(config_fn, 'r') as stream:
    configs = yaml.safe_load(stream)
task_name = configs['task_name_stage_2']

# Initialize W&B
if use_wandb:
    wandb.init(project=configs['project_name'], name = task_name, config=configs)
    wandb.config =configs

# Load data   
stage_2_data = DataloaderStage2(configs['stage_2_data_folder'],  configs['stage_2_zarr_name'], configs)
objects = stage_2_data.get_objects() #((0, 'obj'), (1, 'terrain'), (2, 'robot')) # two objects, one is obj, and one 1 terrain
applied_forces = stage_2_data.get_forces()
total_time_steps = stage_2_data.get_N_time_steps()
time_stamps = stage_2_data.get_time_stamps()
robot_poses, robot_linear_vels = stage_2_data.get_robot_poses_vels()

# calculate GT obj displacements 
gt_obj_poses = stage_2_data.get_obj_poses()
gt_obj_pos = np.array(gt_obj_poses[-1][-3:]) - np.array(gt_obj_poses[0][-3:])
gt_obj_rot = np.array(so3.error(gt_obj_poses[0][0:9], gt_obj_poses[-1][0:9]))

if not os.path.exists(configs['stage_2_save_dir']):
    os.makedirs(configs['stage_2_save_dir'])
with open(configs['stage_2_save_dir'] + 'configs.yaml', 'w') as file:
    yaml.dump(configs, file)

def force_func(t):
    if t < 0:
        return None
    if t >= time_stamps[-1] + configs['dt']:
        f = [0]*6
    elif t >= time_stamps[-1]:
        f = [0]*3 + list(applied_forces[-1])
    else:
        for i in range(len(time_stamps) - 1):
            if t >= time_stamps[i] and t < time_stamps[i+1]:
                f = [0]*3 + list(applied_forces[i])
                break
    only_y_f = [0]*6
    only_y_f[3:5] = f[3:5]
    return torch.from_numpy(np.array(only_y_f)).to(device)

world = SaPWorld(configs, objects)

# load stage 1 results, and initialize robot
world.initialize_from_stage_1(configs['stage_1_dir'], (np.array(robot_poses[0][-3:]), \
                            np.array([1,0,0,0]), np.zeros(6)), reset_mass_fric=True,\
                            reset_robot_fric=True)
# apply control force
world.apply_force(2, force_func)

# Rendering parameters
bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device=device)
projection_matrix = stage_2_data.get_projection_matrix()
camera_matrix = stage_2_data.get_camera_matrix()
fovx, fovy = stage_2_data.get_fovs()

# Initial states for resetting
initial_states = world.get_current_state_dict()
partial_chamfer_distance = PartialChamferLoss(10000)
inertial_start_epoch = configs.get('inertial_start_epoch', 0)
best_epoch = 1
best_loss = 1e6
start_epoch = 1
time_steps = configs['loss_time_steps']

def create_video_from_images(image_list, output_path, fps=30):
    height, width = image_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in image_list:
        bgr_image = image
        out.write(bgr_image)
    out.release()

with torch.autograd.set_detect_anomaly(True):
    progressBar = tqdm(range(start_epoch, configs['stage_2_epochs']+1), desc="Training progress")
    for epoch in range(start_epoch, configs['stage_2_epochs']+1):
        world.zero_grad()
        torch.cuda.empty_cache()
        world.set_train()

        # Get data
        gt_rgb, gt_depth, gtMask, viewMatrix, pcd, pcd_mask = stage_2_data[0]
        pcd_gt = torch.from_numpy(pcd[pcd_mask==1])[:,0:3].to(device).unsqueeze(0)

        # Reset world state after each epoch
        world.set_state_from_dict(initial_states)
        world.reset()
        world.set_state_from_dict(initial_states)
        world.apply_force(2, force_func)

        pred_obj_pos0 = world.objects[1].pos.cpu().detach().numpy()
        pred_obj_rot0 = world.objects[1].rot.cpu().detach().numpy() 

        world.update_learning_rate(epoch, inertial_start_epoch)
        loss = 0
        counter = 0
        
        ## optimize shape and geometry in the first frame
        # convert pybullet depth to actually units
        near, far = stage_2_data.get_near_far_planes()
        gt_depth = far * near / (far - (far - near) *gt_depth)
        if configs['color_representation'] == 'point_color':
            cam = convert_camera_pytorch3D(0, viewMatrix, projection_matrix, gt_rgb, device, near, far, fovy*180/np.pi)
        elif configs['color_representation'] == 'GS':
            cam = convert_camera(0, viewMatrix, fovx, fovy, torch.from_numpy(gt_rgb), device)
    
        rgb, depth, obj_mask, _ = world.render(cam, background, device)
        if configs['color_representation'] == 'point_color':
            gt_rgb = transform(gt_rgb).float().to(device)
            rgb = torch.transpose(rgb, 0, 2)
            rgb = torch.transpose(rgb, 1, 2)
        elif configs['color_representation'] == 'GS':
            gt_rgb = transform(gt_rgb).float().to(device)

        # rgb loss
        mask = (gtMask != 2)
        mask= torch.from_numpy(mask).to(device)
        mask = mask.unsqueeze(0).expand_as(rgb)
        rgb = rgb*mask + 1 * (~mask)
        gt_rgb = gt_rgb*mask + 1 * (~mask)
        Ll1 = l1_loss(rgb, gt_rgb)
        rgb_loss_0 =  (1.0 - configs['lambda_dssim']) * Ll1 + configs['lambda_dssim'] * (1.0 - ssim(rgb, gt_rgb))
        
        # chamfer loss
        v,f,n,_ = world.objects[1].get_mesh()
        chamfer_loss_0 = partial_chamfer_distance(pcd_gt, v) 
        loss += configs['frame_0_chamfer_weight']*chamfer_loss_0*(len(time_steps) + 1)

        # depth loss
        gt_depth = torch.from_numpy(gt_depth).float().to(device)
        mask = (gtMask != 2)
        mask_tensor = torch.from_numpy(mask).to(device)
        gt_depth = gt_depth * mask_tensor
        gt_depth = gt_depth.unsqueeze(0)

        depth = depth * mask_tensor
        depth = depth.unsqueeze(0)
        Ll1 = l1_loss(depth, gt_depth)         
        depth_loss = Ll1 
        loss += configs['depth_weight']*depth_loss

        if configs.get('log_video',False) and epoch%configs['save_every'] == 0:
            debug_video_frames = []
            rgb, depth, mask, _ = world.render(cam, background, device, render_robot=True)
            rgb = torch.transpose(rgb, 0, 2)
            rgb = torch.transpose(rgb, 1, 2)
            rgb = rgb.detach().cpu().numpy()
            rgb = (np.transpose(rgb, (1, 2, 0))*255).astype(np.uint8)
            debug_video_frames.append(rgb)

        # Optimize for the rest of the trajectory
        dynamics_loss = 0
        while world.get_t() < time_stamps[time_steps[-1]] - 1e-6:
            q, dq, fc = world.simulate_step(post_step_update = True, first_epoch = epoch == 1)
            counter += 1
            if configs.get('log_video',False) and epoch%configs['save_every'] == 0:
                rgb, depth, mask, _ = world.render(cam, background, device, render_robot=True)
                rgb = torch.transpose(rgb, 0, 2)
                rgb = torch.transpose(rgb, 1, 2)
                rgb = rgb.detach().cpu().numpy()
                rgb = (np.transpose(rgb, (1, 2, 0))*255).astype(np.uint8)
                debug_video_frames.append(rgb)

            if counter in time_steps:
                gt_rgb, gt_depth, gtMask, viewMatrix, pcd, pcd_mask = stage_2_data[counter]
                gt_pcd = torch.from_numpy(pcd[pcd_mask==1])[:,0:3].to(device).unsqueeze(0)

                # Unilateral chamfer loss
                v,f,n,_ = world.objects[1].get_mesh()
                chamfer_loss = partial_chamfer_distance(gt_pcd, v)
                loss += configs['chamfer_weight']*chamfer_loss

                # robot pos loss
                robot_pos_loss = mse(torch.tensor(robot_poses[counter][-3:]).to(device).float(),world.objects[2].pos.float())
                loss += configs['robot_error_weight']*robot_pos_loss

        # robot errors
        robot_y_error =  math.fabs(robot_poses[counter][-2] - world.objects[2].pos.detach().cpu().numpy()[-2])
        robot_x_error =  math.fabs(robot_poses[counter][-3] - world.objects[2].pos.detach().cpu().numpy()[-3])

        # calculate obj errors 
        pred_obj_pos = world.objects[1].pos.cpu().detach().numpy() - pred_obj_pos0
        pred_obj_rot = so3.error(so3.from_quaternion(pred_obj_rot0),
                                so3.from_quaternion(list(world.objects[1].rot.cpu().detach().numpy()))) 
        obj_pos_loss = np.linalg.norm(gt_obj_pos - pred_obj_pos)
        obj_rot_loss = np.linalg.norm(so3.error(so3.from_moment(gt_obj_rot), so3.from_moment(pred_obj_rot)))

        ## rgb loss for ground
        gt_rgb, gt_depth, gtMask, viewMatrix, pcd, pcd_mask = stage_2_data[-1]
        gt_rgb = transform(gt_rgb).float().to(device)
        rgb, depth, mask, _ = world.render(cam, background, device)
        rgb = torch.transpose(rgb, 0, 2)
        rgb = torch.transpose(rgb, 1, 2)
        rgb_clone = rgb.clone()
        gt_rgb_clone = gt_rgb.clone()
        rgb_clone = rgb_clone*(1 -  mask.unsqueeze(0).expand_as(rgb_clone))
        gt_rgb_clone = gt_rgb_clone*(1 -  mask.unsqueeze(0).expand_as(gt_rgb_clone))
        mask_tensor = torch.from_numpy((gtMask == 0)).expand(3, -1, -1).to(device)
        rgb_clone *= mask_tensor
        gt_rgb_clone *= mask_tensor
        Ll1 = l1_loss(rgb_clone, gt_rgb_clone)
        rgb_loss_last =  (1.0 - configs['lambda_dssim']) * Ll1 + configs['lambda_dssim'] * (1.0 - ssim(rgb_clone, gt_rgb_clone))
        rgb_loss = rgb_loss_0 + rgb_loss_last
        loss += configs['rgb_weight']*rgb_loss
        
        # reg loss
        mesh = world.objects[1].get_mesh()
        pytorch3d_mesh = pytorch3dMesh(verts = mesh[0].float(), faces = mesh[1].float(), \
                    verts_normals = mesh[2].float())
        reg_loss = mesh_laplacian_smoothing(pytorch3d_mesh)
        loss += reg_loss*configs['smooth_reg']

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            for i in range(len(world.objects)):
                obj = world.objects[i]
                obj.save(os.path.join(configs['stage_2_save_dir'],f'obj_{obj.ID}_full.pth'))
            world.objects[1].log_mesh(configs['stage_2_save_dir'],f'epoch_{epoch}')

        fn = os.path.join(configs['stage_2_save_dir'],f'GT_rgb.png')
        if not os.path.isfile(fn):
            gt_rgb = gt_rgb.detach().cpu().numpy()
            gt_rgb = np.transpose(gt_rgb, (1, 2, 0))*255
            cv2.imwrite(fn, gt_rgb)

        try:
            loss.backward()
            world.step(optimize_sap = True, optimize_inertial= epoch >= inertial_start_epoch)
        except Exception as e: 
            print(e)
            cprint.warn('Simulator failed')

        # ensure that the color grid is not OOB
        if hasattr(world.objects[1],'color_grid'):
            world.objects[1].color_grid = world.objects[1].color_grid.clamp(0,1)
        if hasattr(world.objects[0],'colors'):
            world.objects[0].colors = world.objects[0].colors.clamp(0,1)

        if use_wandb:
            log_dict = {"epoch": epoch, "loss": loss.item(), 
                    'chamfer_loss':chamfer_loss.item(), 
                    'robot_x_mae':robot_x_error,
                    'robot_y_mae':robot_y_error,
                    'robot_pos_loss': robot_pos_loss.item(),
                    'obj1_fric':world.objects[1].fric_coeff.item(),
                    'obj1_mass':world.objects[1].mass.item(),
                    'best_epoch': best_epoch,
                    'rgb_loss': rgb_loss.item(),
                    'reg_loss': reg_loss.item(),
                    'best_loss':best_loss,
                    'com_x': world.objects[1].com[0].item(),
                    'com_y': world.objects[1].com[1].item(),
                    'com_z': world.objects[1].com[2].item(),
                    'scale_x': world.objects[1].inertia_scale[0].item(),
                    'scale_y': world.objects[1].inertia_scale[1].item(),
                    'scale_z': world.objects[1].inertia_scale[2].item()}
            log_dict['obj_pos_loss'] = obj_pos_loss
            log_dict['obj_rot_loss'] = obj_rot_loss
            wandb.log(log_dict)

        with torch.no_grad():
            world.set_eval()
            if epoch % 1 == 0:
                progressBar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progressBar.update(1)
            if epoch == configs['stage_2_epochs']:
                progressBar.close()
            
            if epoch%configs['save_every'] == 0:
                depth = depth.detach().cpu().numpy()
                depth *= 255
                depth = depth.astype(np.uint8).squeeze()
                cv2.imwrite(os.path.join(configs['stage_2_save_dir'], f'depth_{epoch}.png'), depth)
                rgb = rgb.detach().cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))*255
                cv2.imwrite(os.path.join(configs['stage_2_save_dir'], f'rgb_{epoch}.png'), rgb)
                if configs.get('log_video',False):
                    create_video_from_images(debug_video_frames, os.path.join(configs['stage_2_save_dir'], f'video_{epoch}.mp4'), fps = 10)

        world.objects[1].com = world.objects[1].com.clamp(-0.05,0.05)
        world.objects[1].inertia_scale = world.objects[1].inertia_scale.clamp(0.5,1.5)
        # ensure that mass and friction parameters are nonnegative
        if 'friction' in configs['optimizable_phy_param']:
            for id in configs['optimizable_phy_ids']:
                if world.objects[id].fric_coeff < configs['min_fric']:
                    world.objects[id].fric_coeff = torch.tensor(configs['min_fric']).to(device)
        if 'mass' in configs['optimizable_phy_param']:
            for id in configs['optimizable_phy_ids']:
                if world.objects[id].mass < configs['min_mass']:
                    world.objects[id].mass = torch.tensor(configs['min_mass']).to(device)                    

wandb.finish()

