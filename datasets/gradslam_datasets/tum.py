import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils import data
from natsort import natsorted

from .basedataset import GradSLAMDataset

class TUMDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        pose_reference_sequence: Optional[str] = None,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        self.pose_reference_sequence = pose_reference_sequence
        self.pose_reference_transform = None
        self.pose_reference_map0_abs = None
        self.pose_reference_query0_abs = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        if self.pose_reference_sequence is not None:
            reference_folder = os.path.join(basedir, self.pose_reference_sequence)
            map0_abs = self._first_associated_pose(reference_folder)
            query0_abs = torch.from_numpy(self._first_associated_pose(self.input_folder)).to(dtype=torch.float64)
            map0_abs = torch.from_numpy(map0_abs).to(dtype=torch.float64)
            transform = torch.linalg.inv(map0_abs) @ query0_abs
            if self.relative_pose:
                self.transformed_poses = transform.to(self.transformed_poses) @ self.transformed_poses
            else:
                self.transformed_poses = torch.linalg.inv(map0_abs).to(self.transformed_poses) @ self.transformed_poses
            self.pose_reference_transform = transform.cpu().numpy()
            self.pose_reference_map0_abs = map0_abs.cpu().numpy()
            self.pose_reference_query0_abs = query0_abs.cpu().numpy()

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.str_, skiprows=skiprows, comments='#')
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def _first_associated_pose(self, input_folder):
        pose_list = os.path.join(input_folder, 'groundtruth.txt')
        if not os.path.isfile(pose_list):
            pose_list = os.path.join(input_folder, 'pose.txt')
        image_data = self.parse_list(os.path.join(input_folder, 'rgb.txt'))
        depth_data = self.parse_list(os.path.join(input_folder, 'depth.txt'))
        pose_data = self.parse_list(pose_list, skiprows=1)
        associations = self.associate_frames(
            image_data[:, 0].astype(np.float64),
            depth_data[:, 0].astype(np.float64),
            pose_data[:, 0].astype(np.float64),
        )
        if not associations:
            raise ValueError(f"No associated RGB-D/GT frames in {input_folder}")
        return self.pose_matrix_from_quaternion(pose_data[associations[0][2], 1:].astype(np.float64))

    def get_filepaths(self):

        frame_rate = 32
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
            pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.input_folder, 'pose.txt')):
            pose_list = os.path.join(self.input_folder, 'pose.txt')

        image_list = os.path.join(self.input_folder, 'rgb.txt')
        depth_list = os.path.join(self.input_folder, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        color_paths, depth_paths = [], []
        for ix in indicies:
            (i, j, k) = associations[ix]
            color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
            depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]

        embedding_paths = None

        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        
        frame_rate = 32
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
            pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.input_folder, 'pose.txt')):
            pose_list = os.path.join(self.input_folder, 'pose.txt')

        image_list = os.path.join(self.input_folder, 'rgb.txt')
        depth_list = os.path.join(self.input_folder, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        color_paths, poses, depth_paths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
            depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return poses
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)
    
