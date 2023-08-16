import sys
sys.path.append('../code')
import os
import torch
import argparse
import datetime
from pathlib import Path
from pyhocon import ConfigFactory
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import BackprojectDepth
from model.loss import compute_scale_and_shift

def get_plot_data(model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
    batch_size, num_samples, _ = rgb_gt.shape

    rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
    normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
    normal_map = (normal_map + 1.) / 2.
    
    depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
    depth_gt = depth_gt.to(depth_map.device)
    scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
    depth_map = depth_map * scale + shift
    
    # save point cloud
    depth = depth_map.reshape(1, 1, img_res[0], img_res[1])
    pred_points = get_point_cloud(depth, model_input, model_outputs)

    gt_depth = depth_gt.reshape(1, 1, img_res[0], img_res[1])
    gt_points = get_point_cloud(gt_depth, model_input, model_outputs)
    
    plot_data = {
        'rgb_gt': rgb_gt,
        'normal_gt': (normal_gt + 1.)/ 2.,
        'depth_gt': depth_gt,
        'pose': pose,
        'rgb_eval': rgb_eval,
        'normal_map': normal_map,
        'depth_map': depth_map,
        "pred_points": pred_points,
        "gt_points": gt_points,
    }

    return plot_data

def get_point_cloud(depth, model_input, model_outputs):
    img_res = [384, 512]
    backproject = BackprojectDepth(1, img_res[0], img_res[1]).cuda()

    color = model_outputs["rgb_values"].reshape(-1, 3)
    
    K_inv = torch.inverse(model_input["intrinsics"][0])[None]
    points = backproject(depth, K_inv)
    pose = torch.inverse(model_input["pose"][0])[None]
    points = pose @ points
    points = points[0, :3, :].permute(1, 0)
    points = torch.cat([points, color], dim=-1)
    return points.detach().cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    opt = parser.parse_args()

    GPU_INDEX = opt.local_rank
    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1    
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    torch.distributed.barrier()

    data_dir = 'scannet'
    root_dir = "../exps/"
    conf = '../code/confs/scannet_mlp.conf'
    exp_name = "scannet_mlp"
    out_dir = "evaluation/scannet_mlp"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scenes = ["scene0710_00", "scene0710_01"]
    all_results = []
    for idx, scan in enumerate(scenes):
        idx = 2
        cur_exp = f"{exp_name}_{idx}"
        cur_root = os.path.join(root_dir, cur_exp)
        # use first timestamps
        dirs = sorted(os.listdir(cur_root))
        cur_root = os.path.join(cur_root, dirs[0])
        model_path = os.path.join(cur_root, "checkpoints/ModelParameters/latest.pth")
        config = ConfigFactory.parse_file(conf)

        img_res = [384, 512]
        test_dataset = utils.get_class(config.get_string('train.dataset_class'))(data_dir, img_res, idx, center_crop_type = 'no_crop', mode='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       collate_fn=test_dataset.collate_fn,
                                                       )
        
        
        conf_model = config.get_config('model')
        model = utils.get_class(config.get_string('train.model_class'))(conf=conf_model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        saved_model_state = torch.load(model_path)
        model.load_state_dict(saved_model_state["model_state_dict"])
        model.eval()

        psnrs = []
        it = iter(test_dataloader)
        for i in range(len(test_dataset)):
            indices, model_input, ground_truth = next(it)
            print(indices)
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()
            
            total_pixels = test_dataset.total_pixels
            split_n_pixels = config.get_int('train.split_n_pixels', default=10000)
            split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
            res = []
            for s in tqdm(split):
                out = model(s, indices)
                d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'depth_values': out['depth_values'].detach()}
                if 'rgb_un_values' in out:
                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                res.append(d)

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            plot_data = get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])
            
            plots_dir = os.path.join(cur_root, 'test')
            utils.mkdir_ifnotexists(plots_dir)
            plot_conf = config.get_config('plot')
            plt.plot(model.module.implicit_network,
                    indices,
                    plot_data,
                    plots_dir,
                    'test',
                    img_res,
                    **plot_conf
                    )
            
            psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                      ground_truth['rgb'].cuda().reshape(-1,3))
            psnrs.append(psnr)
        
        with open(f'{plots_dir}/psnr_list.txt', 'w') as f:
            for item in psnrs:
                f.write(str(item) + '\n')
        print('AC!')
        sys.exit(0)
            