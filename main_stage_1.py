import os
import argparse

import numpy as np
import yaml
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes as pytorch3dMesh
from icecream import ic
from tqdm import tqdm
import torch 
import torchvision.transforms as transforms
import cv2
from cprint import *
import wandb
import open3d as o3d
import faulthandler
faulthandler.enable()

from diffworld.data import DataloaderStage1
from diffworld.world import SaPWorld
from diffworld.utils import convert_camera_pytorch3D,  l1_loss, mse, ssim, PartialChamferLoss, psnr

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Script for running stage 1 training")
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
task_name = configs['task_name_stage_1']

# Initialize W&B
if use_wandb:
    wandb.init(project=configs['project_name'], name = task_name, config=configs)

# Load data    
stage_1_data = DataloaderStage1(configs['stage_1_data_folder'], configs['stage_1_zarr_name'], configs)
objects = stage_1_data.get_objects() #((0, 'obj'), (1, 'terrain'), (2, 'robot'))
pcds, pcd_masks, visible_pcds, visible_pcd_masks = stage_1_data.get_pcd()

# Initialize World
world = SaPWorld(configs, objects[0:2])
world.initialize(pcds, pcd_masks)
# Make sure rgb values stay within range
world.objects[1].color_grid = world.objects[1].color_grid.clamp(0,1)

if not os.path.exists(configs['save_dir']):
    os.makedirs(configs['save_dir'])
with open(configs['save_dir'] + 'configs.yaml', 'w') as file:
    yaml.dump(configs, file)

# Rendering parameters
bg_color = [1, 1, 1] 
background = torch.tensor(bg_color, dtype=torch.float32, device=device)
projection_matrix = stage_1_data.get_projection_matrix()
camera_matrix = stage_1_data.get_camera_matrix()
fovx, fovy = stage_1_data.get_fovs()

# Initial states for resetting
initial_states = world.get_current_state_dict()

# Original sap points for regularization
original_sap = world.objects[1].sap_inputs.clone()

# Training setup
partial_chamfer_distance = PartialChamferLoss(10000)
pcd_gt = torch.from_numpy(visible_pcds[visible_pcd_masks==1])[:,0:3].to(device).unsqueeze(0)
best_epoch = 1
best_loss = 1e6

with torch.autograd.set_detect_anomaly(True):
    progressBar = tqdm(range(1, configs['stage_1_epochs']+1), desc="Training progress")
    for epoch in range(1, configs['stage_1_epochs']+1):
        world.zero_grad()
        world.set_train()

        # Get data
        gt_rgbs, gt_depths, gt_masks, view_matrices = stage_1_data.get_all_data()
        idx = 0 #first time step only
        gt_depth, gt_rgb, view_matrix, gt_mask = gt_depths[idx],gt_rgbs[idx], view_matrices[idx], gt_masks[idx]
        
        # Reset world state after each epoch
        if not (epoch == 1):
            world.set_state_from_dict(initial_states)
            world.reset()
            world.set_state_from_dict(initial_states)

        # convert pybullet depth to actually units
        near, far = stage_1_data.get_near_far_planes()
        gt_depth = far * near / (far - (far - near) *gt_depth)
        cam = convert_camera_pytorch3D(idx, view_matrix, projection_matrix, gt_rgb, device, near, far, fovy*180/np.pi)

        world.update_learning_rate(epoch)
        loss = 0

        rgb, depth, obj_mask, _ = world.render(cam, background, device)
        gt_rgb = transform(gt_rgb).float().to(device)
        rgb = torch.transpose(rgb, 0, 2)
        rgb = torch.transpose(rgb, 1, 2)

        # mask out robot
        mask = (gt_mask != 2)
        rgb_mask_tensor = torch.from_numpy(mask).to(device)
        rgb_mask_tensor = rgb_mask_tensor.unsqueeze(0).expand_as(rgb)
        rgb = rgb*rgb_mask_tensor + 1 * (~rgb_mask_tensor)
        gt_rgb = gt_rgb*rgb_mask_tensor + 1 * (~rgb_mask_tensor)

        # rgb loss
        Ll1 = l1_loss(rgb, gt_rgb)
        rgb_loss =  (1.0 - configs['lambda_dssim']) * Ll1 + configs['lambda_dssim'] * (1.0 - ssim(rgb, gt_rgb))
        loss += configs['rgb_weight']*rgb_loss
        
        # sap regularization loss
        sap_reg_loss = l1_loss(world.objects[1].sap_inputs, original_sap)
        loss += configs['sap_reg']*sap_reg_loss

        # unilateral chamfer loss
        v,f,n,_ = world.objects[1].get_mesh()
        chamfer_loss = partial_chamfer_distance(pcd_gt, v) 
        loss += configs['chamfer_weight']*chamfer_loss

        gt_depth = torch.from_numpy(gt_depth).float().to(device)

        # mask out robot
        mask_tensor = torch.from_numpy(mask).to(device)
        gt_depth = gt_depth*mask_tensor
        gt_depth = gt_depth.unsqueeze(0)
        depth = depth*mask_tensor
        depth = depth.unsqueeze(0)

        # depth rendering loss
        Ll1 = l1_loss(depth, gt_depth)         
        depth_loss = (1.0 - configs['lambda_dssim_depth']) * Ll1 + \
                configs['lambda_dssim_depth'] * (1.0 - ssim(depth, gt_depth))
        loss += configs['depth_weight']*depth_loss

        physics_loss_wandb = 0
        physics_loss = 0
        contacts = world.simulator.contacts
        penetration_depths = []
        for contact in contacts:
            penetration_depths.append(contact[0][3])
        # penetration loss
        if len(penetration_depths)>= 1:
            penetration_depths = torch.cat(penetration_depths).squeeze(0).to(device).float()
            physics_loss += l1_loss(penetration_depths, torch.zeros_like(penetration_depths))

        # log object mesh
        if epoch%configs['save_every'] == 0:
            world.objects[1].log_mesh(configs['save_dir'],f'obj_mesh_{epoch}')
        
        # static balance loss
        balance_loss = 0
        init_q = world.objects[1].p.clone().detach()

        if configs['static_balance_sim_steps'] > 0:
            for iter in range(configs['static_balance_sim_steps']):
                q, dq, _ = world.simulate_step()
                balance_loss += l1_loss(q[4:], init_q[4:]) 
            balance_loss /= configs['static_balance_sim_steps']
            physics_loss += balance_loss

        loss += configs['static_balance_weight']*physics_loss
        if physics_loss > 0:
            physics_loss_wandb = physics_loss.item()

        # smoothing
        mesh = world.objects[1].get_mesh()
        pytorch3d_mesh = pytorch3dMesh(verts = mesh[0].float(), faces = mesh[1].float(), \
                    verts_normals = mesh[2].float())
        reg_loss = mesh_laplacian_smoothing(pytorch3d_mesh)
        loss += reg_loss*configs['smooth_reg']

        # rendering metric
        with torch.no_grad():
            rgb_mse = mse(gt_rgb, rgb)
            rgb_ssim = ssim(gt_rgb, rgb)
            rgb_psnr = psnr(gt_rgb, rgb).mean()
            depth_mse = mse(gt_depth, depth)
            depth_ssim = ssim(gt_depth, depth)
        
        # save gt images for debugging
        fn = os.path.join(configs['save_dir'],f'GT_rgb.png')
        if not os.path.isfile(fn):
            gt_rgb = gt_rgb.detach().cpu().numpy()
            gt_rgb = np.transpose(gt_rgb, (1, 2, 0))*255
            cv2.imwrite(fn, gt_rgb)
            gt_depth = gt_depth.detach().cpu().numpy()
            gt_depth *= 255
            gt_depth = gt_depth.astype(np.uint8).squeeze()
            cv2.imwrite(os.path.join(configs['save_dir'],f'GT_depth.png'), gt_depth)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            # log best model
            for i in range(len(world.objects)):
                obj = world.objects[i]
                world.objects[i].save(os.path.join(configs['save_dir'],f'obj_{obj.ID}_best.pth'))
                world.objects[1].log_mesh(configs['save_dir'],f'best')

        # backward all losses
        try:
            world.objects[1].p.retain_grad()
            world.objects[1].psr_grid.retain_grad()
            loss.backward()
            world.step()
        except Exception as e:
            # occasionaly the simulator could fail due to numerical issues 
            cprint.warn(e)
            cprint.warn('Simulator failed')

        with torch.no_grad():
            world.set_eval()
            # logging and postprocessing
            if epoch % 1 == 0:
                progressBar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progressBar.update(1)
            if epoch == configs['stage_1_epochs']:
                progressBar.close()
            if epoch%configs['save_every'] == 0:
                rgb = rgb.detach().cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))*255
                cv2.imwrite(os.path.join(configs['save_dir'], f'rgb_{epoch}.png'), rgb)
                depth = depth.detach().cpu().numpy()
                depth *= 255
                depth = depth.astype(np.uint8).squeeze()
                cv2.imwrite(os.path.join(configs['save_dir'], f'depth_{epoch}.png'), depth)
                obj_mask = obj_mask.cpu().numpy()
                obj_mask = (obj_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(configs['save_dir'], f'mask_{epoch}.png'), obj_mask)

            # clamp colors to be valid
            if hasattr(world.objects[1],'color_grid'):
                world.objects[1].color_grid = world.objects[1].color_grid.clamp(0,1)
            if hasattr(world.objects[0],'colors'):
                world.objects[0].colors = world.objects[0].colors.clamp(0,1)

        if use_wandb:
            wandb.log({"epoch": epoch, "loss": loss.item(), 
                    'chamfer_loss':chamfer_loss.item(),\
                    'rgb_loss':rgb_loss.item(),\
                    'physics loss:':physics_loss_wandb,\
                    'depth_loss':depth_loss.item(),\
                    'rgb_mse': rgb_mse.item(),\
                    'rgb_ssim': rgb_ssim.item(),\
                    'rgn_psnr': rgb_psnr.item(),\
                    'depth_mse': depth_mse.item(),\
                    'depth_ssim': depth_ssim.item(),\
                    'reg_loss': reg_loss.item(),\
                    'sap_reg_loss': sap_reg_loss.item(),\
                    'best_epoch': best_epoch})
wandb.finish()

