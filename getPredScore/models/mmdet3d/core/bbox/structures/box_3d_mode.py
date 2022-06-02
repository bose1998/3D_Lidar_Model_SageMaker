# Copyright (c) OpenMMLab. All rights reserved.
from enum import IntEnum, unique
import json
import numpy as np
import torch

from .base_box3d import BaseInstance3DBoxes
from .cam_box3d import CameraInstance3DBoxes
from .depth_box3d import DepthInstance3DBoxes
from .lidar_box3d import LiDARInstance3DBoxes
from .utils import limit_period


@unique
class Box3DMode(IntEnum):
    r"""Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is [0.5, 1.0, 0.5],
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth mode:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(box, src, dst, rt_mat=None, with_yaw=True):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.ndarray |
                torch.Tensor | :obj:`BaseInstance3DBoxes`):
                Can be a k-tuple, k-list or an Nxk array/tensor, where k = 7.
            src (:obj:`Box3DMode`): The src Box mode.
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.
            with_yaw (bool, optional): If `box` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.

        Returns:
            (tuple | list | np.ndarray | torch.Tensor |
                :obj:`BaseInstance3DBoxes`):
                The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'Box3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        if is_Instance3DBoxes:
            with_yaw = box.with_yaw

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if with_yaw:
            yaw = arr[..., 6:7]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                yaw = -yaw + np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                yaw = yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        else:
            raise NotImplementedError(
                f'Conversion from Box3DMode {src} to {dst} '
                'is not supported yet')

        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[..., :3] @ rt_mat.t()

        if with_yaw:
            remains = arr[..., 7:]
            arr = torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)
        else:
            remains = arr[..., 6:]
            arr = torch.cat([xyz[..., :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            #print("after yaw change")
            #print(arr)
            with open('models/samp.json') as file1:
              data=json.load(file1)
            #print(result[0]['pts_bbox']['boxes_3d'].tensor[0])
            #print(len(data['annotations']))
            li=[]
            coun=0
            for i in range(len(arr)):
              save=arr[i]
              #data['annotations'][i]['label']='car'
              #li.append({'id': '6214a7eb6937efeeffc34162', 'datasetId': '61df2c877e42622a71758961', 'type': 'cube_3d', 'label': 'Vehicle.Truck', 'attributes': [], 'metadata': {'system': {'status': None, 'startTime': 0, 'endTime': 200, 'frame': 0, 'endFrame': 200, 'snapshots_': [], 'parentId': None, 'clientId': 'ffbf0b57-bd11-4b95-b370-124bf893a44f', 'automated': False, 'objectId': '7', 'isOpen': False, 'isOnlyLocal': False, 'frameNumberBased': True, 'attributes': {}, 'clientParentId': None, 'system': False, 'description': None, 'itemLinks': [], 'openAnnotationVersion': '1.32.3-prod.8', 'recipeId': '61d6a9d1090fd05f635b17d0'}, 'user': {}}, 'creator': 'avi@dataloop.ai', 'createdAt': '2022-02-22T09:07:55.919Z', 'updatedBy': 'diwakar.negi@tcs.com', 'updatedAt': '2022-02-24T05:02:30.337Z', 'itemId': '61e06e0467146c2a1bfd3b9b', 'url': 'https://gate.dataloop.ai/api/v1/annotations/6214a7eb6937efeeffc34162', 'item': 'https://gate.dataloop.ai/api/v1/items/61e06e0467146c2a1bfd3b9b', 'dataset': 'https://gate.dataloop.ai/api/v1/datasets/61df2c877e42622a71758961', 'hash': '317a9d67fc54c955a079ce2e7c931ff456793d21', 'source': 'ui', 'coordinates': {'position': {'x': -10.264884948730469, 'y': -13.781776428222656, 'z': -0.2731616497039795}, 'scale': {'x': 2.8543715476989746, 'y': 12.071895599365234, 'z': 3.6031875610351562}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 1.6209666942046397e-09}}})
              #li[coun]['coordinates']['position']['x']=float(save[1])
              #li[coun]['coordinates']['position']['y']=-float(save[0])
              #li[coun]['coordinates']['position']['z']=float(save[2])
              #li[coun]['coordinates']['scale']['x']=float(save[3])
              #li[coun]['coordinates']['scale']['y']=float(save[4])
              #li[coun]['coordinates']['scale']['z']=float(save[5])
              #li[coun]['coordinates']['rotation']['x']=float(0.0)
              #li[coun]['coordinates']['rotation']['y']=float(0.0)
              data['annotations'][coun]['coordinates']['rotation']['z']=float(save[6]) - np.pi/2
              coun=coun+1
            #data['annotations']=li
            with open('models/samp.json','w') as outfile:
              json.dump(data,outfile)
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type}'
                    ' is not supported yet')
            return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw)
        else:
            return arr
