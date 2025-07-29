import os
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box ,LidarPointCloud
from timemoseg.utils.tools import ( gen_dx_bx, get_nusc_maps)
from nuscenes.utils.geometry_utils import view_points
from timemoseg.utils.geometry import (
    update_intrinsics,
    calculate_bev_params,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose)


 

def get_scene_list(file_path):
    #print(file_path)  
    scene_list = []
    with open(file_path, 'r') as split_file:
        for line in split_file:
            scene_list.append(line.replace('\n', ''))
    return scene_list



def depth_augmentation(cam_depth, crop, resize=0.3, resize_dims=(224 , 480),  ):   
    

    H, W = resize_dims   
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]  # 
    cam_depth[:, 1] -= crop[1]  
    

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0


    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)   
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map) 


class MOSDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 # 
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.is_train = is_train
        self.cfg = cfg
        # if 
        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

         
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD # 3
        self.train_scene = get_scene_list( cfg.TRAIN_SCENE_PATH)   
        self.val_scene = get_scene_list( cfg.VAL_SCENE_PATH)  # val scene
        self.test_scene = get_scene_list( cfg.TEST_SCENE_PATH)  # test scene

        self.scenes =  self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()
        #  
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()  

        #  
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )  #  

        # 
        bev_resolution, bev_start_position, bev_dimension = calculate_bev_params(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        ) 
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # 
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])  

        
        self.nusc_maps = get_nusc_maps(self.cfg.DATASET.MAP_FOLDER)  
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
         

    def get_scenes(self):        
        blacklist = [419] + self.nusc_can.can_blacklist  #
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]        
        if self.is_train ==0:
            scenes= self.train_scene
        elif self.is_train ==1:
            scenes= self.val_scene
        else: 
            scenes= self.test_scene

        for scene_no in blacklist: #
            if scene_no in scenes: #  
                scenes.remove(scene_no)   
        
        return scenes
    
        

    def prepro(self):
        samples = [samp for samp in self.nusc.sample] 
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.receptive_field):
                index_t = index + t
                #
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # 
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break
                
                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE  #  
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP  
        crop_w = int(max(0, (resized_width - final_width) / 2))
        #  
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_lidar_depth(self, lidar_points,  lidar_ego_pose, lidar_calibrated_sensor, cam_ego_pose, cam_calibrated_sensor, img = [1600, 900]):
        
        pts_img, depth = self.map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)  #
        
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)


    def get_input_data(self, rec):
        
        images = []
        yuyi_prori = [] 
        intrinsics = []
        extrinsics = []
        came_file_name=[]
        cameras = self.cfg.IMAGE.NAMES  
        depths=[]
        # 
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:  
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            #  
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world  
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()  



            if self.cfg.IMAGE.USEDEPTH:
                lidar_path = lidar_sample['filename'] # 
                lidar_points = np.fromfile(os.path.join(
                    self.dataroot, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                lidar_calibrated_sensor = self.nusc.get( 'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
                
                point_depth = self.get_lidar_depth(  lidar_points, 
                       lidar_pose, lidar_calibrated_sensor, car_egopose, sensor_sample)   
             
                depth_images = depth_augmentation( point_depth ,crop=self.augmentation_parameters['crop'] ) #  
                depths.append(depth_images.unsqueeze(0).unsqueeze(0))
            
            
            # Load  
            image_filename = os.path.join(self.dataroot, camera_sample['filename']) # 
            img = Image.open(image_filename)
            came_file_name.append( image_filename)
         
            normalised_img = self.normalise_image(img)  

            #
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            # load  
            base_root = os.path.join(self.dataroot, 'Semantic_3cat')  #
            filename = os.path.basename(camera_sample['filename']).split('.')[0] + '.png'  #  
            yuyi_image_filename = os.path.join(base_root, cam, 'semantic_map', filename)
            yuyi_img = Image.open(yuyi_image_filename)
            # 
            yuyi_img = np.asarray( yuyi_img ).astype('float32')
            normalised_yuyi_img = torch.tensor(np.transpose(yuyi_img, (2,0,1))/255.0 )


            normalised_img= torch.concat((normalised_img, normalised_yuyi_img))
            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            yuyi_prori.append(normalised_yuyi_img.unsqueeze(0).unsqueeze(0) )  # 
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))
            

        images, intrinsics, extrinsics , yuyi_prori= (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1),
                                          torch.cat( yuyi_prori, dim=1),
                                          ) #  
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)
             
        return images, intrinsics, extrinsics, came_file_name ,depths, yuyi_prori

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    
    def get_birds_eye_view_label(self, rec, instance_map, ): # 
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))  
        moving_seg = np.zeros((self.bev_dimension[0], self.bev_dimension[1])) #  
            
        for annotation_token in rec['anns']:   
            #  
            annotation = self.nusc.get('sample_annotation', annotation_token)
            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1  :
                continue  # 
            
            # NuScenes filter
            if 'vehicle' in annotation['category_name']:
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
                if 'cb5118da1ab342aa947717dc53544259' in annotation['attribute_tokens'] :  
                    cv2.fillPoly(moving_seg, [poly_region], 1.0) # 
            elif 'human' in annotation['category_name']:
                
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
                if 'ab83627ff28b465b85c427162dec722f' in annotation['attribute_tokens'] :
                    cv2.fillPoly(moving_seg, [poly_region], 1.0)    
        
        
        return segmentation, moving_seg  

    def map_pointcloud_to_image(self, 
        lidar_points,  
        img,  
        lidar_calibrated_sensor,
        lidar_ego_pose,
        cam_calibrated_sensor,
        cam_ego_pose,
        min_dist: float = 0.0,
    ):

        lidar_points = LidarPointCloud(lidar_points.T)
        lidar_points.rotate(
            Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))  

        
        lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_ego_pose['translation']))
        lidar_points.translate(-np.array(cam_ego_pose['translation']))
        lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

        
        lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
        lidar_points.rotate(
            Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

        depths = lidar_points.points[2, :]
        coloring = depths  
        points = view_points(lidar_points.points[:3, :],
                            np.array(cam_calibrated_sensor['camera_intrinsic']),
                            normalize=True)

        
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < img[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < img[1] - 1)
        points = points[:, mask] # point
        coloring = coloring[mask]

        return points, coloring


    
    
    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, instance_map, ):
        segmentation_np, moving_seg_np  = \
            self.get_birds_eye_view_label(rec, instance_map )  # 
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0) #  
        # 
        moving_seg = torch.from_numpy( moving_seg_np ).long().unsqueeze(0).unsqueeze(0)
        
        
        return segmentation, moving_seg
        

    def get_time_egomotion(self, rec, index):
        rec_t0 = rec
        egomotion = np.eye(4, dtype=np.float32) 

        if index < len(self.ixes) - 1:  
            rec_t1 = self.ixes[index + 1] 

            if rec_t0['scene_token'] == rec_t1['scene_token']:  
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                egomotion[3, :3] = 0.0
                egomotion[3, 3] = 1.0

        egomotion = torch.Tensor(egomotion).float()

        # 
        egomotion = mat2pose_vec(egomotion)
        return egomotion.unsqueeze(0)

    
    def voxelize_hd_map(self, rec):  
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND) 
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]] # [50.0, 50.0]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']] 

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # 
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2) 
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        ) #  (200, 200) 

        elements = self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS  
        hd_features = self.nusc_maps[map_name].get_map_mask(box_coords, rot * 180 / np.pi , elements, canvas_size=canvas_size)

        hd_features = torch.from_numpy(hd_features[None]).float()  
        hd_features = torch.transpose(hd_features,-2,-1)  
        return hd_features  

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 
                'egomotion', 'hdmap',   'indices','moving_seg',  'segmentation', 'depths',
                'filename','scene' ,"semantic_info",   ]  
        for key in keys:
            data[key] = []

        instance_map = {}
       

        for i, index_t in enumerate(self.indices[index]):
            
            rec = self.ixes[index_t]
            # print( 
            scene_sample = self.nusc.get('scene', rec["scene_token"])
            scene_name = scene_sample['name']
            data['scene'].append(scene_name)
            #  
            if i < self.receptive_field: #
                images, intrinsics, extrinsics, filename_img, depths, yuyi_img = self.get_input_data(rec) #  
                data['image'].append(images) #  
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
                data['filename'].append( filename_img)
                data['depths'].append(depths)
                data['semantic_info'].append(yuyi_img)
                
            
            segmentation ,moving_seg  = self.get_label(rec, instance_map )

            egomotion = self.get_time_egomotion(rec, index_t) 
            hd_map_feature = self.voxelize_hd_map(rec)  

            data['segmentation'].append(segmentation) 
            
            data['egomotion'].append(egomotion)
            data['hdmap'].append(hd_map_feature)
            data['indices'].append(index_t) 
            data['moving_seg'].append(moving_seg)  
            
        # 
        for key, value in data.items():  #
            if key in ['image', 'intrinsics', 'extrinsics',  'egomotion', 'segmentation','hdmap', 'moving_seg', 'depths', 'semantic_info']:
                if key == 'depths' and self.cfg.IMAGE.USEDEPTH  is False:
                    continue
                if key == 'filename' :
                    continue
                if key == 'scene' :
                    continue
                data[key] = torch.cat(value, dim=0) 
        
        return data