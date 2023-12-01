# This code was prepared by Xueqian Li and Chaoyang Wang for their work on NTP-CVPR'22.


import os
import glob
import copy
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import argoverse.data_loading.object_label_record as object_label
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.cuboid_interior import filter_point_cloud_to_bbox_3D_vectorized
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat


def create_se3(obj):
    return SE3(
        rotation=quat2rotmat(obj.quaternion), translation=np.array(obj.translation)
    )


class PreProcessArgoverseDataset(Dataset):
    def __init__(
        self,
        dataset_path="",
        partition="val",
        remove_ground=True,
        get_gt_tracks=True,
        max_correspondence_distance=1.0,
        width=50,
        compensate_egomotion=False,
    ):
        self.compensate_egomotion = compensate_egomotion
        self.partition = partition
        self.width = width

        self.remove_ground = remove_ground
        self.get_gt_tracks = get_gt_tracks
        self.max_correspondence_distance = max_correspondence_distance

        if self.partition == "train":
            self.datapath = sorted(glob.glob(f"{dataset_path}/training/*/*/lidar"))
        elif self.partition == "test":
            self.datapath = sorted(glob.glob(f"{dataset_path}/testing/*/*/lidar"))
        elif self.partition == "val":
            self.datapath = sorted(glob.glob(f"{dataset_path}/val/*/lidar"))
        print(f"Found {len(self.datapath)} data files.")
        self.avm = ArgoverseMap()

    def __getitem__(self, index):
        filename = self.datapath[index]
        print(f"Processing {filename}")
        log_id = filename.split("/")[-2]
        dataset_dir = filename.split(log_id)[0]
        city_info_fpath = f"{dataset_dir}/{log_id}/city_info.json"
        city_info = read_json_file(city_info_fpath)
        city_name = city_info["city_name"]

        # ANCHOR: Get consecutive point clouds.
        lidar_sweep_fnames = sorted(glob.glob(f"{dataset_dir}{log_id}/lidar/PC_*"))
        lidar_sweep_idx = 0

        lidar_sweeps = get_lidar_sweeps(lidar_sweep_fnames, lidar_sweep_idx, self.width)

        if self.remove_ground:
            for i, lidar_sweep in enumerate(lidar_sweeps):
                pc_t = lidar_sweep.pose.transform_point_cloud(lidar_sweep.ply)
                _, not_ground_logicals = self.avm.remove_ground_surface(
                    copy.deepcopy(pc_t), city_name, return_logicals=True
                )
                pc_t = pc_t[not_ground_logicals]
                lidar_sweeps[i].ply = lidar_sweep.pose.inverse_transform_point_cloud(
                    pc_t
                ).astype("float32")

        # NOTE: remove non drivable area
        drivable_area = False
        if drivable_area:
            for i, lidar_sweep in enumerate(lidar_sweeps):
                pc_t = lidar_sweep.pose.transform_point_cloud(lidar_sweep.ply)
                pc_t = self.avm.remove_non_driveable_area_points(pc_t, city_name)
                lidar_sweeps[i].ply = lidar_sweep.pose.inverse_transform_point_cloud(
                    pc_t
                ).astype("float32")

        # NOTE: Remove points above certain height.
        max_height = 4.0
        for i, lidar_sweep in enumerate(lidar_sweeps):
            indices = np.where(lidar_sweeps[i].ply[:, 2] <= max_height)[0]
            lidar_sweeps[i].ply = lidar_sweeps[i].ply[indices]

        # NOTE: Remove points beyond certain distance.
        max_dist = 80
        for i, lidar_sweep in enumerate(lidar_sweeps):
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(lidar_sweeps[i].ply)
            dists_to_center = np.sqrt(np.sum(lidar_sweeps[i].ply ** 2, 1))
            ind = np.where(dists_to_center <= max_dist)[0]
            lidar_sweeps[i].ply = lidar_sweeps[i].ply[ind]

        # ANCHOR: get trajectory annotation for points in the first frame
        if self.get_gt_tracks:
            n1 = lidar_sweeps[0].ply.shape[0]
            tracks1_fpath = f"{dataset_dir}{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{lidar_sweeps[0].timestamp}.json"

            objects1 = object_label.read_label(tracks1_fpath)

            # NOTE: parse object annotation files for each consecutive frames
            objects2_track_id_dict_list = []
            for i in range(1, self.width):
                tracks_other_fpath = f"{dataset_dir}{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{lidar_sweeps[i].timestamp}.json"
                objects_other = object_label.read_label(tracks_other_fpath)
                objects2_track_id_dict_list.append(
                    {x.track_id: x for x in objects_other}
                )

            print(tracks1_fpath)

            traj = np.zeros((n1, self.width, 3), dtype="float32")
            traj_val_mask = np.zeros((n1, self.width), dtype="float32")

            # NOTE: iterate over consecutive frames
            for i in range(1, self.width):
                mask1_tracks_flow = []
                mask2_tracks_flow = []

                for object1 in objects1:
                    if object1.occlusion == 100:
                        continue

                    # NOTE: add a margin to the cuboids. Some cuboids are very tight and might lose some points.
                    margin = 0.6
                    object1.length = object1.length + margin
                    object1.height = object1.height + margin
                    object1.width = object1.width + margin
                    bbox1_3d = object1.as_3d_bbox()
                    # ego[0]
                    inbox_pc1, is_valid = filter_point_cloud_to_bbox_3D_vectorized(
                        bbox1_3d, lidar_sweeps[0].ply
                    )
                    indices = np.where(is_valid == True)[0]
                    mask1_tracks_flow.append(indices)

                    full_mask1 = np.arange(len(lidar_sweeps[0].ply))

                    objects2_track_id_dict = objects2_track_id_dict_list[i - 1]

                    if object1.track_id in objects2_track_id_dict:
                        object2 = objects2_track_id_dict[object1.track_id]
                        # Assuming points are not moving within the object:
                        # obj[i+1] == obj[i] == obj
                        # ego[i] <- obj
                        obj_pose1 = create_se3(object1)
                        # ego[i+1] <- obj
                        obj_pose2 = create_se3(object2)
                        # ego[i+1] <- obj <- ego[i]
                        relative_pose_1_2 = obj_pose2.right_multiply_with_se3(
                            obj_pose1.inverse()
                        )
                        # obj points from sweep i in the obj's frame from the sweep i+1
                        inbox_pc1_t = relative_pose_1_2.transform_point_cloud(inbox_pc1)

                        translation = inbox_pc1_t - inbox_pc1
                        traj[indices, i, :] = translation
                        traj_val_mask[
                            indices, i
                        ] = 1  # NOTE: mark the corresponding points as valid on trajectory.
                        bbox2_3d = object2.as_3d_bbox()
                        _, is_valid2 = filter_point_cloud_to_bbox_3D_vectorized(
                            bbox2_3d, lidar_sweeps[i].ply
                        )
                        mask2_tracks_flow.append(np.where(is_valid2 == True)[0])
                    else:
                        traj_val_mask[
                            indices, i
                        ] = 0  # NOTE: mark the points without associations with zero

                # ANCHOR: Compensate egomotion to get rigid flow.
                map_relative_to_base = lidar_sweeps[i].pose  # city <- ego[i]
                map_relative_to_other = lidar_sweeps[0].pose  # city <- ego[0]
                # ego[i] <- city <- ego[0]
                other_to_base = map_relative_to_base.inverse().right_multiply_with_se3(
                    map_relative_to_other
                )
                points = lidar_sweeps[0].ply
                # points from sweep #0 in the coordinate frame of sweep #i.
                points_t = other_to_base.transform_point_cloud(points)

                mask1_tracks_flow = np.unique(np.hstack(mask1_tracks_flow))
                mask1_no_tracks = np.setdiff1d(
                    full_mask1, mask1_tracks_flow, assume_unique=True
                )

                mask2_tracks_flow = np.unique(np.hstack(mask2_tracks_flow))

                # ANCHOR: refine the rigid registration with ICP (without the tracks)
                # NOTE: this might be unnecessary since the given ego pose is pretty good already

                full_mask2 = np.arange(len(lidar_sweeps[i].ply))

                mask2_no_tracks = np.setdiff1d(
                    full_mask2, mask2_tracks_flow, assume_unique=True
                )

                # # points from sweep #0 in the coordinate frame of sweep #i.
                # pc1_o3d = o3d.geometry.PointCloud()
                # pc1_o3d.points = o3d.utility.Vector3dVector(points_t[mask1_no_tracks])
                # # points from sweep #i in the coordinate frame of sweep #i.
                # pc2_o3d = o3d.geometry.PointCloud()
                # pc2_o3d.points = o3d.utility.Vector3dVector(
                #     lidar_sweeps[i].ply[mask2_no_tracks]
                # )

                # # ANCHOR: apply point-to-point ICP
                # trans_init = np.identity(4)
                # reg_p2p = o3d.pipelines.registration.registration_icp(
                #     pc1_o3d,
                #     pc2_o3d,
                #     self.max_correspondence_distance,
                #     trans_init,
                #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                #     o3d.pipelines.registration.ICPConvergenceCriteria(
                #         max_iteration=100
                #     ),
                # )
                # pc1_t_o3d = pc1_o3d.transform(reg_p2p.transformation)
                # points_t_refined = np.asarray(pc1_t_o3d.points)
                # rigid_flow = points_t_refined - points[mask1_no_tracks]

                rigid_flow = points_t[mask1_no_tracks] - points[mask1_no_tracks]
                traj[mask1_no_tracks, i, :] = rigid_flow
                traj_val_mask[mask1_no_tracks, i] = 1

            # ANCHOR: annotate forward flow
            print("start annotating forward scene flow....")

            flows = []
            mask1_tracks_flow_list = []
            mask2_tracks_flow_list = []

            # ANCHOR: iterate over consecutive frames
            for i in range(0, self.width - 1):
                mask1_tracks_flow = []
                mask2_tracks_flow = []

                n1 = lidar_sweeps[i].ply.shape[0]
                tracks1_fpath = f"{dataset_dir}{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{lidar_sweeps[i].timestamp}.json"
                tracks2_fpath = f"{dataset_dir}{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{lidar_sweeps[i+1].timestamp}.json"

                objects1 = object_label.read_label(tracks1_fpath)
                objects2 = object_label.read_label(tracks2_fpath)
                objects2_track_id_dict = {
                    object2.track_id: object2 for object2 in objects2
                }

                flow = np.zeros((n1, 3), dtype="float32")
                flow_val_mask = np.zeros(n1, dtype="float32")

                for object1 in objects1:
                    if object1.occlusion == 100:
                        continue

                    if object1.track_id in objects2_track_id_dict:
                        object2 = objects2_track_id_dict[object1.track_id]
                        # Assuming points are not moving within the object:
                        # obj[i+1] == obj[i] == obj
                        # ego[i] <- obj
                        obj_pose1 = create_se3(object1)
                        # ego[i+1] <- obj
                        obj_pose2 = create_se3(object2)
                        # ego[i+1] <- obj <- ego[i]
                        relative_pose_1_2 = obj_pose2.right_multiply_with_se3(
                            obj_pose1.inverse()
                        )

                        # NOTE: add a margin to the cuboids. Some cuboids are very tight and might lose some points.
                        margin = 0.6
                        object1.length = object1.length + margin
                        object1.height = object1.height + margin
                        object1.width = object1.width + margin
                        bbox1_3d = object1.as_3d_bbox()
                        inbox_pc1, is_valid = filter_point_cloud_to_bbox_3D_vectorized(
                            bbox1_3d, lidar_sweeps[i].ply
                        )
                        indices = np.where(is_valid == True)[0]
                        mask1_tracks_flow.append(indices)
                        inbox_pc1_t = relative_pose_1_2.transform_point_cloud(inbox_pc1)

                        translation = inbox_pc1_t - inbox_pc1
                        flow[indices, :] = translation
                        flow_val_mask[
                            indices
                        ] = 1  # NOTE: mark the corresponding points as valid on trajectory.

                        bbox2_3d = object2.as_3d_bbox()
                        _, is_valid2 = filter_point_cloud_to_bbox_3D_vectorized(
                            bbox2_3d, lidar_sweeps[i + 1].ply
                        )
                        mask2_tracks_flow.append(np.where(is_valid2 == True)[0])
                    else:
                        flow_val_mask[
                            indices
                        ] = 0  # NOTE: mark the flow without object associations with zero

                # ANCHOR: compensate egomotion to get rigid flow.
                # city <- ego[i+1]
                map_relative_to_base = lidar_sweeps[i + 1].pose
                # city <- ego[i]
                map_relative_to_other = lidar_sweeps[i].pose
                # ego[i+1] <- city <- ego[i]
                other_to_base = map_relative_to_base.inverse().right_multiply_with_se3(
                    map_relative_to_other
                )
                # ego[i]
                points = lidar_sweeps[i].ply
                # Points from #i in ego[i+1]
                points_t = other_to_base.transform_point_cloud(points)

                mask1_tracks_flow = np.unique(np.hstack(mask1_tracks_flow))
                mask2_tracks_flow = np.unique(np.hstack(mask2_tracks_flow))

                # ANCHOR: refine the rigid registration with ICP (without the tracks)
                # NOTE: this might be unnecessary since the given ego pose is pretty good already
                full_mask1 = np.arange(len(lidar_sweeps[i].ply))
                full_mask2 = np.arange(len(lidar_sweeps[i + 1].ply))
                mask1_no_tracks = np.setdiff1d(
                    full_mask1, mask1_tracks_flow, assume_unique=True
                )
                mask2_no_tracks = np.setdiff1d(
                    full_mask2, mask2_tracks_flow, assume_unique=True
                )

                pc1_o3d = o3d.geometry.PointCloud()
                # Points from #i in ego[i+1]
                pc1_o3d.points = o3d.utility.Vector3dVector(points_t[mask1_no_tracks])
                pc2_o3d = o3d.geometry.PointCloud()
                # Points from #i+1 in ego[i+1]
                pc2_o3d.points = o3d.utility.Vector3dVector(
                    lidar_sweeps[i + 1].ply[mask2_no_tracks]
                )

                # ANCHOR: apply point-to-point ICP
                trans_init = np.identity(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pc1_o3d,
                    pc2_o3d,
                    self.max_correspondence_distance,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=100
                    ),
                )
                pc1_t_o3d = pc1_o3d.transform(reg_p2p.transformation)
                points_t_refined = np.asarray(pc1_t_o3d.points)

                # The rigid flow is in ego[i+1] frame
                rigid_flow = points_t_refined - points[mask1_no_tracks]
                flow[mask1_no_tracks] = rigid_flow
                flow_val_mask[mask1_no_tracks] = 1  # mark rigid flow as valid
                flows.append(flow)
                mask1_tracks_flow_list.append(mask1_tracks_flow)
                mask2_tracks_flow_list.append(mask2_tracks_flow)

        ego_poses = []
        for lidar_sweep in lidar_sweeps:
            # city <- ego@t=i
            ego_poses.append(lidar_sweep.pose.transform_matrix)
        if self.compensate_egomotion:
            pcs_city = []
            flows_city = []
            for i, flow in enumerate(flows):
                # sweep #i at timestamp t0=i
                lidar_sweep_t0 = lidar_sweeps[i]
                # sweep #i+1 at timestamp t1=i+1
                lidar_sweep_t1 = lidar_sweeps[i + 1]
                # Converts from the coordinate frame of the sweep #i into city's coordinate frame.
                # city <- ego@t0
                city_from_ego_t0 = lidar_sweep_t0.pose
                # city <- ego@t1
                city_from_ego_t1 = lidar_sweep_t1.pose
                # ego@t0 <- ego@t1
                ego_t0_from_t1 = city_from_ego_t0.inverse().right_multiply_with_se3(
                    city_from_ego_t1
                )
                # Points from the sweep #i+1 in the coordinate frame of the sweep #i
                points_f0_t0_ego = lidar_sweep_t0.ply
                # NOTE: The original `flow` is in the ego@t1 coordinate frame
                points_f0_t1_ego = (
                    points_f0_t0_ego + ego_t0_from_t1.transform_point_cloud(flow)
                )
                points_f0_t0_city = city_from_ego_t0.transform_point_cloud(
                    points_f0_t0_ego
                )
                points_f0_t1_city = city_from_ego_t0.transform_point_cloud(
                    points_f0_t1_ego
                )
                pcs_city.append(points_f0_t0_city)
                # NOTE: Motion compensated `flow` is in the city coordinate frame
                flows_city.append(points_f0_t1_city - points_f0_t0_city)
            pcs = pcs_city
            flows = flows_city
        else:
            # NOTE: `flows` has scene flow with egomotion (no compensation).
            pcs = [lidar_sweep.ply for lidar_sweep in lidar_sweeps]

        sample = dict(
            log_id=log_id,
            pcs=pcs,
            flows=flows,
            traj=traj,
            traj_val_mask=traj_val_mask,
            mask1_tracks_flow=mask1_tracks_flow_list,
            mask2_tracks_flow=mask2_tracks_flow_list,
            flow_val_mask=flow_val_mask,
            poses=ego_poses,
        )
        return sample

    def __len__(self):
        return len(self.datapath)


@dataclass
class PlyWithPose:
    """Struct to hold ply and pose data."""

    ply: np.ndarray
    pose: SE3
    timestamp: int


def get_lidar_sweeps(
    base_directory: Path, sweep_index: int, width: int
) -> Optional[List[PlyWithPose]]:
    """Get the lidar sweep from the given sweep_directory.
    ​
        Args:
            sweep_directory: path to middle lidar sweep.
            sweep_index: index of the middle lidar sweep.
            width: +/- lidar scans to grab.
        Returns:
            List of plys with their associated pose if all the sweeps exist.
    """
    sweeps = []

    n_forward = width
    start = sweep_index
    end = start + n_forward
    if start < 0:
        start = 0
    if end >= len(base_directory):
        end = len(base_directory) - 1

    for step_index in range(start, end + 1):
        sweep_path = base_directory[step_index]
        ply = load_ply(sweep_path)

        if ply is None:
            return None

        ply_timestamp = sweep_path.split("PC_")[1].split(".")[0]
        log_path = base_directory[0].split("lidar")[0]
        pose_fname = glob.glob(
            f"{log_path}/poses/city_SE3_egovehicle_{ply_timestamp}.json"
        )[0]
        pose_ = read_json_file(pose_fname)
        pose = SE3(
            rotation=quat2rotmat(pose_["rotation"]),
            translation=np.array(pose_["translation"]),
        )

        timestamp = int(sweep_path.split("/")[-1].split("_")[1][:-4])
        if pose is None:
            return None

        bundle = PlyWithPose(ply, pose, timestamp)
        sweeps.append(bundle)

    return sweeps


if __name__ == "__main__":
    dataset_path = "/mnt/local/argoverse/argoverse-tracking/"
    output_dir = "/mnt/local/argoverse/av1_traj_w_pose_no_egomotion"
    os.makedirs(output_dir, exist_ok=True)

    partition = "val"
    max_correspondence_distance = 1.0

    dataset = PreProcessArgoverseDataset(
        dataset_path=dataset_path,
        partition=partition,  # 'train' or 'val' only
        remove_ground=True,
        get_gt_tracks=True,
        max_correspondence_distance=max_correspondence_distance,
        compensate_egomotion=True,
    )
    for i, sample in enumerate(dataset):
        print(f"Sample #{i}")
        saved_path = os.path.join(output_dir, sample["log_id"])
        np.savez_compressed(saved_path, **sample)
