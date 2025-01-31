# Pairwise scene flow estimation with MBNSF.

import os, glob
import sys
import argparse
import logging
import csv
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.general_utils import *
from utils.nsfp_utils import *
from utils.o3d_uitls import extract_clusters_dbscan
from utils.sc_utils import spatial_consistency_loss

logger = logging.getLogger(__name__)

def sceneflow_deep_prior(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    info_dict: dict,
    options: argparse.Namespace
):

    net = Neural_Prior(filter_size=options.hidden_units).to(options.device).eval()

    if options.backward_flow:
        net_inv = Neural_Prior(filter_size=options.hidden_units).to(options.device).eval()
    
    # ANCHOR: Initialize network with meta prior
    if info_dict:
        net.load_state_dict(info_dict["state_dict_forward"])
        if options.backward_flow:
            net_inv.load_state_dict(info_dict["state_dict_backward"])
    
    if options.backward_flow:
        params = [{'params': net.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay},
                {'params': net_inv.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay}]
    else:
        params = net.parameters()
        
    if options.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay)
    elif options.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=options.lr, weight_decay=options.weight_decay)

        
    # ANCHOR: Initialize optimizer
    if info_dict:
        optimizer.load_state_dict(info_dict["optimizer_state_dict"])

    early_stopping = EarlyStopping(patience=options.early_patience, min_delta=options.early_min_delta)

    if options.time:
        timers = Timers()
        timers.tic("solver_timer")

    # Extract cluster masks:
    labels = extract_clusters_dbscan(pc1, eps = options.sc_cluster_eps, min_points=options.sc_cluster_min_points, return_clusters= False, return_colored_pcd=False)
    labels_t = torch.from_numpy(labels).float().clone().to(options.device)
    label_ids = torch.unique(labels_t)[1:] # Ignore unclustered points with label = -1.
    num_clusters = len(label_ids)
    assert num_clusters > 0

    pc1 = pc1.to(options.device).contiguous()
    pc2 = pc2.to(options.device).contiguous()

    # ANCHOR: initialize best metrics
    best_loss_1 = 100.
    best_epoch = 0
    best_flow = None
    
    for epoch in range(options.iters):
        optimizer.zero_grad()

        flow_pred_1 = net(pc1)
        pc1_deformed = pc1 + flow_pred_1

        # Chamfer Distance Loss
        loss_chamfer_1, _ = my_chamfer_fn(pc2.unsqueeze(0), pc1_deformed.unsqueeze(0), None, None)

        # Spatial Consistency Loss
        loss_sc = torch.zeros([1,1], dtype=pc1_deformed.dtype, device=pc1_deformed.device,)
        for id in label_ids:
            cluster_ids = labels_t == id
            num_cluster_points = torch.count_nonzero(cluster_ids)
            if num_cluster_points > 2:
                cluster = pc1[cluster_ids]
                cluster_deformed = pc1_deformed[cluster_ids]
                assert cluster.shape == cluster_deformed.shape
                cluster_cs_loss = spatial_consistency_loss(cluster.unsqueeze(0), cluster_deformed.unsqueeze(0), d_thre=options.sc_dist_thresh)
                loss_sc += cluster_cs_loss
        loss_sc /= num_clusters
        loss_sc = loss_sc.squeeze()
        loss_sc = options.sc_loss_weight * loss_sc
        
        if options.backward_flow:
            flow_pred_1_prime = net_inv(pc1_deformed)
            pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
            loss_chamfer_1_prime, _ = my_chamfer_fn(pc1_prime_deformed.unsqueeze(0), pc1.unsqueeze(0), None, None)
        
        if options.backward_flow:
            loss_chamfer = loss_chamfer_1 + loss_chamfer_1_prime
        else:
            loss_chamfer = loss_chamfer_1

        loss = loss_chamfer + loss_sc

        flow_pred_1_final = pc1_deformed - pc1
        
        # ANCHOR: get best metrics
        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_epoch = epoch
            best_flow = flow_pred_1_final
            
        if epoch % 50 == 0:
            logging.info(f"[Ep: {epoch}] [Loss: {loss:.5f}],   [loss_chamfer: {loss_chamfer:.5f}][loss_sc: {loss_sc:.5f}] ")

        if early_stopping.step(loss):
            logging.info(f"Early Stop: [Ep: {epoch}] [Loss: {loss:.5f}],   [loss_chamfer: {loss_chamfer:.5f}][loss_sc: {loss_sc:.5f}] ")
            break

        loss.backward()
        optimizer.step()

    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        logging.info(timers.print())

    # ANCHOR: get the best metrics
    info_dict = {
        'loss': best_loss_1,
        'time': time_avg,
        'epoch': best_epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'state_dict_forward': net.state_dict()
    }
    
    if options.backward_flow:
        info_dict_inv = {
            'state_dict_backward': net_inv.state_dict()
        }
        info_dict.update(info_dict_inv)
        
    flow_pred = best_flow.detach().cpu()
        
    return flow_pred, labels, info_dict


def fit_sequence_of_scene_flow_field(
    exp_dir,
    pc_list,
    options,   
    flow_gt_list = None,
    ):

    csv_file = open(f"{exp_dir}/metrics.csv", 'w')
    metric_labels = ['epe', 'acc_strict', 'acc_relax', 'angle_error', 'outlier', 'time']
    csv_writer = csv.DictWriter(csv_file, metric_labels)
    csv_writer.writeheader()

    n_lidar_sweeps = len(pc_list)

    cur_metrics = {}
    for label in metric_labels:
        cur_metrics[label] = np.zeros(n_lidar_sweeps-1)
    flow_out = []
    labels_out = []
    for i in range(n_lidar_sweeps-1):
        pc1 = torch.from_numpy(pc_list[i]).float().clone()
        pc2 = torch.from_numpy(pc_list[i + 1]).float().clone()

        if i == 0:
            info_dict = None
        else:
            # Initialize with the previous deep prior weights. Likely that the scene flow will be similar.
            info_dict = info_dict_forward

        logger.info(f"{i}->{i + 1}")

        # ANCHOR: Run scene flow estimation
        flow_pred, labels, info_dict_forward = sceneflow_deep_prior(pc1, pc2, info_dict, options)
        flow_out.append(flow_pred)
        labels_out.append(labels)

        # evaluate flow metrics
        EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(flow_pred.unsqueeze(0),
            torch.from_numpy(flow_gt_list[i]).unsqueeze(0))
        logging.info(f" [EPE: {EPE3D_1:.3f}] [Acc strict: {acc3d_strict_1 * 100:.3f}%]"
                    f" [Acc relax: {acc3d_relax_1 * 100:.3f}%] [Angle error (rad): {angle_error_1:.3f}]"
                    f" [Outl.: {outlier_1 * 100:.3f}%]")
        cur_metrics['epe'][i] = EPE3D_1
        cur_metrics['acc_strict'][i] = acc3d_strict_1
        cur_metrics['acc_relax'][i] = acc3d_relax_1
        cur_metrics['angle_error'][i] = angle_error_1
        cur_metrics['outlier'][i] = outlier_1
        cur_metrics['time'][i] = info_dict_forward['time']

        # save info_dict
        torch.save(info_dict_forward["state_dict_forward"],  f"{exp_dir}/model_{i}.pth")

    cur_metrics = {label:round(cur_metrics[label].mean(), 4) for label in cur_metrics.keys()}
    csv_writer.writerow(cur_metrics)
    logging.info(cur_metrics)


    return flow_out, labels_out, cur_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise scene flow estimation with MBNSF.")
    parser.add_argument('--exp_name', type=str, default='fit_MBNSF_tl25_lr3_ep100', metavar='N', help='Name of the experiment.')
    parser.add_argument('--iters', type=int, default=1000, metavar='N', help='Number of iterations to optimize the model.')#5000
    parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam', 'lbfgs', 'lbfgs_custm', 'rmsprop'), help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='Learning rate.')#0.001
    parser.add_argument('--momentum', type=float, default=0, metavar='M', help='SGD momentum (default: 0.9).')
    parser.add_argument('--device', default='cuda:0', type=str, help='device: cpu? cuda?')
    parser.add_argument('--dataset_path', type=str, default='/mnt/088A6CBB8A6CA742/av1/av1_traj', metavar='N', help='Dataset path.')
    parser.add_argument('--time', dest='time', action='store_true', default=True, help='Count the execution time of each step.')
    parser.add_argument('--traj_len', type=int, default=25, help='point cloud sequence length for the trajectory.')
    
    # For neural prior
    parser.add_argument('--weight_decay', type=float, default=0, metavar='N', help='Weight decay.')
    parser.add_argument('--hidden_units', type=int, default=128, metavar='N', help='Number of hidden units in neural prior')
    parser.add_argument('--layer_size', type=int, default=8, help='how many hidden layers in the model.')
    parser.add_argument('--act_fn', type=str, default='relu', metavar='AF', help='activation function for neural prior.')
    parser.add_argument('--backward_flow', action='store_true', default=True, help='use backward flow or not.')
    parser.add_argument('--early_patience', type=int, default=100, help='patience in early stopping.')#100
    parser.add_argument('--early_min_delta', type=float, default=0.0001, help='the minimum delta of early stopping.')

    # For spatial consistency regularizer
    parser.add_argument('--sc_loss_weight', type=float, default=1.0, help='weigth for spatial consistency loss.')
    parser.add_argument('--sc_dist_thresh', type=float, default=0.03, help='distance threshold for SC')
    parser.add_argument('--sc_cluster_eps', type=float, default=0.8, help='Epsilon parameter of DBSCAN when extracting clusters.')
    parser.add_argument('--sc_cluster_min_points', type=int, default=30, help='minimum number of points per cluster in DBSCAN')
    
    options = parser.parse_args()

    exp_dir_path = f"checkpoints/{options.exp_name}"
    exp_dir_path = os.path.join(os.path.dirname(__file__), '../', f"checkpoints/{options.exp_name}")
    os.makedirs(exp_dir_path, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{exp_dir_path}/run.log"), logging.StreamHandler()])        

    logging.info('\n' + ' '.join([sys.executable] + sys.argv))
    logging.info(options)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        logging.info(key+': '+str(print_options[key]))
    logging.info('---------------------------------------')

    # load point cloud sequeence
    npz_files = sorted(glob.glob(os.path.join(options.dataset_path, '*.npz')))
    pred_dir = f'{options.dataset_path}_{options.exp_name}_pred'
    os.makedirs(pred_dir, exist_ok=True)

    metric_labels = ['epe', 'acc_strict', 'acc_relax', 'angle_error', 'outlier', 'time']
    seq_metrics = {}
    for label in metric_labels:
        seq_metrics[label] = np.zeros(len(npz_files))

    for fi_id, fi_name in enumerate(npz_files):
        logging.info(f"ID: {fi_id}/{len(npz_files)}")
        log_id = fi_name.split('/')[-1].split('.')[0]
    
        data = np.load(fi_name, allow_pickle=True)

        pc_list = [data['pcs'][i] for i in range(options.traj_len)]
        flow_gt_list = [data['flows'][i] for i in range(options.traj_len-1)]

        cur_exp_dir = os.path.join(exp_dir_path, log_id)
        os.makedirs(cur_exp_dir, exist_ok=True)
        
        flow, labels, metrics = fit_sequence_of_scene_flow_field(cur_exp_dir, pc_list, options, flow_gt_list)
        pred_path = f'{pred_dir}/{log_id}_flow.npz'
        print(f'Store output into {pred_path}')
        np.savez(pred_path,
                 flows=np.asarray([t.numpy() for t in flow], dtype='object'),
                 labels=np.asarray([t for t in labels], dtype='object'))
        seq_metrics['epe'][fi_id] = metrics['epe']
        seq_metrics['acc_strict'][fi_id] = metrics['acc_strict']
        seq_metrics['acc_relax'][fi_id] = metrics['acc_relax']
        seq_metrics['angle_error'][fi_id] = metrics['angle_error']
        seq_metrics['outlier'][fi_id] = metrics['outlier']
        seq_metrics['time'][fi_id] = metrics['time']
    seq_metrics_mean = {label:round(seq_metrics[label].mean(), 4) for label in seq_metrics.keys()}
    logging.info('---------------------------------------')
    logging.info('Final SF Metrics')
    logging.info(seq_metrics_mean)
    logging.info('---------------------------------------')