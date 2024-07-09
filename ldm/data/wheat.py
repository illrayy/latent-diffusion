import os
import numpy as np
import cv2
import PIL
import random
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


class wheatBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 p_dorpout=0,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ag_rate=0.8,
                 degradation=None,
                 crop_size=(256,256)
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = flip_p
        self.ag_rate = ag_rate
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.crop_size = crop_size
        self.p_dorpout = p_dorpout
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = cv2.imread(example["file_path_"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cp = image.copy()
        h, w, c = image.shape
        

        y_ref_start = np.random.randint(0, h - self.crop_size[0] + 1)
        x_ref_start = np.random.randint(0, w - self.crop_size[1] + 1)
        reference = image[y_ref_start:y_ref_start + self.crop_size[0], x_ref_start:x_ref_start + self.crop_size[1]]
        reference = self.image_processor(images=reference)['pixel_values'][0]

        
        if random.random() < self.p_dorpout:
            reference = np.zeros_like(reference)
                                                                                                
        
        if random.random()<self.ag_rate:
            #创建正方形的四个顶点坐标和中心坐标
            square_size = 512
            center = (256, 256)
            square_points = np.array([(0, 0), (0, square_size), (square_size, 0), (square_size, square_size)], dtype=np.float32)

            #随机生成旋转角度
            angle = random.uniform(-90, 90)

            #创建顶点旋转矩阵并应用旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_points = cv2.transform(square_points.reshape(1, -1, 2), rotation_matrix)

            y_min = int(np.min(rotated_points[0, :, 1]))
            y_max = int(np.max(rotated_points[0, :, 1]))
            max_scale = square_size/(y_max - y_min)

            #将缩放因子设置为（0.5，max_scale)
            scale_factor = random.uniform(0.6, max_scale)

            # 创建缩放矩阵
            scaling_matrix = np.array([[scale_factor, 0, (1 - scale_factor) * center[0]],
                                    [0, scale_factor, (1 - scale_factor) * center[1]]], dtype=np.float32)

            # 应用缩放
            scaled_points = cv2.transform(rotated_points, scaling_matrix)

            # 将坐标四舍五入为整数
            scaled_points = np.int32(scaled_points)

            transformed_x_min = int(np.min(scaled_points[:, :, 0]))
            transformed_y_min = int(np.min(scaled_points[:, :, 1]))

            #随机生成平移量，并平移
            shift_x = random.randint(-transformed_x_min, transformed_x_min)
            shift_y = random.randint(-transformed_y_min, transformed_y_min)

            scaled_points[:, :, 0] += shift_x
            scaled_points[:, :, 1] += shift_y


            back_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

            # 执行旋转
            rotated_image = cv2.warpAffine(image, back_rotation_matrix, (h, w))
            scaled_points = cv2.transform(scaled_points.reshape(1, -1, 2), back_rotation_matrix)
            scaled_points = np.int32(scaled_points)

            x_max = np.max(scaled_points[:, :, 0])
            x_min = np.min(scaled_points[:, :, 0])
            y_max = np.max(scaled_points[:, :, 1])
            y_min = np.min(scaled_points[:, :, 1])

            image = rotated_image[y_min:y_max, x_min:x_max]
        
        try:
            image = cv2.resize(image, (self.size, self.size))
        except:
            image = cv2.resize(image_cp, (self.size, self.size))
            print("file_path_")
            print(x_max, x_min, y_max, y_min)
            

        if random.random()<self.flip:
            image = cv2.flip(image, 0)
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["reference"] = reference
        return example


class wheatTrain(wheatBase):
    def __init__(self, txt_file="datasets/wheat/val_ldm.txt", data_root="datasets/wheat/target", ag_rate=0.8, **kwargs):
        super().__init__(txt_file=txt_file, data_root=data_root,
                         ag_rate=ag_rate,**kwargs)


class wheatValidation(wheatBase):
    def __init__(self, txt_file="datasets/wheat/val_ldm.txt", data_root="datasets/wheat/target", flip_p=0.,ag_rate=0., **kwargs):
        super().__init__(txt_file=txt_file, data_root=data_root,
                         flip_p=flip_p, ag_rate=ag_rate, **kwargs)
