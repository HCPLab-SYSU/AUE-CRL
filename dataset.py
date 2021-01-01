import os
import copy
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.utils.data as data

def default_loader(path):
    return Image.open(path).convert('RGB')

def get_au_loc(array_, ori_size, new_size):
    '''
    input: 
        np.ndarray shape=(68 , 2) 
    return:
        np.ndarray shape=(12 , 4) 
    '''

    array = copy.deepcopy(array_)

    arr2d = array.transpose()
    arr2d[0,:]=arr2d[0,:]/ori_size*new_size
    arr2d[1,:]=arr2d[1,:]/ori_size*new_size

    region_bbox=[]
    if arr2d.shape[1] == 68:
        region_bbox+=[[arr2d[0,21],arr2d[1,21],arr2d[0,22],arr2d[1,22]]]    # au 1
        region_bbox+=[[arr2d[0,18],arr2d[1,18],arr2d[0,25],arr2d[1,25]]]    # au 2
        region_bbox+=[[(arr2d[0,21]+arr2d[0,22])/2,(arr2d[1,21]+arr2d[1,22])/2,(arr2d[0,19]+arr2d[0,24])/2,(arr2d[1,19]+arr2d[1,24])/2]]    # au 4
        region_bbox+=[[(arr2d[0,37]+arr2d[0,38])/2,(arr2d[1,37]+arr2d[1,38])/2,(arr2d[0,43]+arr2d[0,44])/2,(arr2d[1,43]+arr2d[1,44])/2]]    # au 5 *
        region_bbox+=[[arr2d[0,41],arr2d[1,41],arr2d[0,46],arr2d[1,46]]]    # au 6
        region_bbox+=[[arr2d[0,38],arr2d[1,38],arr2d[0,43],arr2d[1,43]]]    # au 7
        region_bbox+=[[arr2d[0,31],arr2d[1,31],arr2d[0,35],arr2d[1,35]]]    # au 9 *
        region_bbox+=[[arr2d[0,50],arr2d[1,50],arr2d[0,52],arr2d[1,52]]]    # au 10
        region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 12
        # region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 14
        region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 15
        region_bbox+=[[arr2d[0,58],arr2d[1,58],arr2d[0,56],arr2d[1,56]]]    # au 16 *
        region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]    # au 17
        region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 20 *
        region_bbox+=[[arr2d[0,60],arr2d[1,60],arr2d[0,62],arr2d[1,62]]]    # au 23
        region_bbox+=[[arr2d[0,61],arr2d[1,61],arr2d[0,64],arr2d[1,64]]]    # au 24
        region_bbox+=[[arr2d[0,62],arr2d[1,62],arr2d[0,66],arr2d[1,66]]]    # au 25 *
        region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]    # au 26 *
    elif arr2d.shape[1] == 66:
        region_bbox+=[[arr2d[0,21],arr2d[1,21],arr2d[0,22],arr2d[1,22]]]    # au 1
        region_bbox+=[[arr2d[0,18],arr2d[1,18],arr2d[0,25],arr2d[1,25]]]    # au 2
        region_bbox+=[[(arr2d[0,21]+arr2d[0,22])/2,(arr2d[1,21]+arr2d[1,22])/2,(arr2d[0,19]+arr2d[0,24])/2,(arr2d[1,19]+arr2d[1,24])/2]]    # au 4
        # region_bbox+=[[(arr2d[0,37]+arr2d[0,38])/2,(arr2d[1,37]+arr2d[1,38])/2,(arr2d[0,43]+arr2d[0,44])/2,(arr2d[1,43]+arr2d[1,44])/2]]    # au 5 *
        region_bbox+=[[arr2d[0,41],arr2d[1,41],arr2d[0,46],arr2d[1,46]]]    # au 6
        region_bbox+=[[arr2d[0,38],arr2d[1,38],arr2d[0,43],arr2d[1,43]]]    # au 7
        # region_bbox+=[[arr2d[0,31],arr2d[1,31],arr2d[0,35],arr2d[1,35]]]    # au 9 *
        region_bbox+=[[arr2d[0,50],arr2d[1,50],arr2d[0,52],arr2d[1,52]]]    # au 10
        region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 12
        # region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 14
        region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 15
        # region_bbox+=[[arr2d[0,58],arr2d[1,58],arr2d[0,56],arr2d[1,56]]]    # au 16 *
        region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]    # au 17
        # region_bbox+=[[arr2d[0,48],arr2d[1,48],arr2d[0,54],arr2d[1,54]]]    # au 20 *
        region_bbox+=[[arr2d[0,60],arr2d[1,60],arr2d[0,62],arr2d[1,62]]]    # au 23
        region_bbox+=[[arr2d[0,61],arr2d[1,61],arr2d[0,64],arr2d[1,64]]]    # au 24
        # region_bbox+=[[arr2d[0,62],arr2d[1,62],arr2d[0,66],arr2d[1,66]]]    # au 25 *
        # region_bbox+=[[arr2d[0,51],arr2d[1,51],arr2d[0,57],arr2d[1,57]]]    # au 26 *        
    else:
        print("Invalid Landmark Annotations")

    region_array = np.round(np.array(region_bbox))

    return region_array

class MyDataset_EM(data.Dataset):
    def __init__(self, imgs, labels, landmarks, bboxs, flag, needAU, Model, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        
    def __getitem__(self, index):
        img, label, landmark, bbox = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.landmarks[index]), copy.deepcopy(self.bboxs[index])
        ori_img_w, ori_img_h = img.size

        # BoundingBox
        left  = bbox[0]
        upper = bbox[1]
        right = bbox[2]
        lower = bbox[3]

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=upper

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        if self.target_transform is not None:
            label = self.transform(label)

        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
        landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
        landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
        for i in range(au_location.shape[0]):
            for j in range(4):
                if au_location[i,j]<=11:
                    au_location[i,j] = 12 
                if au_location[i,j]>=45: 
                    au_location[i,j] = 44
        
        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        au_location = torch.LongTensor(au_location)

        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)

class MyDataset_AU(data.Dataset):
    def __init__(self, imgs, labels, landmarks, flag, Model, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.Model = Model

    def __getitem__(self, index):
        img, label, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.landmarks[index])
        ori_img_w, ori_img_h = img.size

        left  = np.min(landmark[:,0])
        right = np.max(landmark[:,0])
        upper = np.min(landmark[:,1])
        lower = np.max(landmark[:,1])

        # Added by Xy
        if self.flag=='train':
            enlarge_bbox = True
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            enlarge_bbox = True
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)

        if enlarge_bbox:
            left = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=upper

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()

        # get au location
        landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
        landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        if self.Model in ['ResNet-101', 'ResNet-50']:
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 11:
                        au_location[i,j] = 12 
                    if au_location[i,j]>=45: 
                        au_location[i,j] = 44
        elif self.Model == 'ResNet-18':
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 112)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 23:
                        au_location[i,j] = 24 
                    if au_location[i,j]>=89: 
                        au_location[i,j] = 88

        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))

        if self.target_transform is not None:
            label = self.transform(label)

        au_location = torch.LongTensor(au_location)

        return trans_img, au_location, label

    def __len__(self): 
        return len(self.imgs)

class MyDataset_EM_Artificial(data.Dataset):
    def __init__(self, imgs, labels, landmarks, bboxs, flag, needAU, Model, Experiment, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        self.Experiment = Experiment
        self.PriorKnowledgeTable_Train = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [1.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float)
        self.PriorKnowledgeTable_Test = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float)
        
    def __getitem__(self, index):
        img, label, landmark, bbox = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.landmarks[index]), copy.deepcopy(self.bboxs[index])
        ori_img_w, ori_img_h = img.size

        # Need AU Label
        if self.Experiment=='AU':
            if self.flag=='train':
                label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if label[i] < 0.5:
                        label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif label[i] < 1.0:
                        label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

        # Face Rotation
        # left  = np.min(landmark[:,0])
        # right = np.max(landmark[:,0])
        # upper = np.min(landmark[:,1])
        # lower = np.max(landmark[:,1])

        # BoundingBox
        left  = bbox[0]
        upper = bbox[1]
        right = bbox[2]
        lower = bbox[3]

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        if self.Model==7:
            left  = np.min(landmark[:,0])
            right = np.max(landmark[:,0])
            upper = np.min(landmark[:,1])
            lower = np.max(landmark[:,1])

            enlarge_bbox = False
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=upper

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        # Visualization
        # img_transform = img.resize((crop_img_w ,crop_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))

        if self.target_transform is not None:
            label = self.transform(label)

        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
        landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
        landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        if self.Model in ['ResNet-101', 'ResNet-50']:
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 11:
                        au_location[i,j] = 12 
                    if au_location[i,j]>=45: 
                        au_location[i,j] = 44
        elif self.Model == 'ResNet-18':
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 112)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 23:
                        au_location[i,j] = 24 
                    if au_location[i,j]>=89: 
                        au_location[i,j] = 88

        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        au_location = torch.LongTensor(au_location)
        
        # Need AU Label
        if self.Experiment=='Fuse':
            if self.flag=='train':
                AU_label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if AU_label[i] < 0.5:
                        AU_label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif AU_label[i] < 1.0:
                        AU_label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        AU_label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                AU_label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

            return (trans_img, self.imgs[index]), au_location, (label, AU_label)
        
        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)

class MyDataset_EM_Artificial_Compound(data.Dataset):
    def __init__(self, imgs, labels, landmarks, bboxs, flag, needAU, Model, Experiment, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        self.Experiment = Experiment
        self.PriorKnowledgeTable_Train = np.array([
            [1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],],dtype=np.float)
        self.PriorKnowledgeTable_Test = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],],dtype=np.float)
        
    def __getitem__(self, index):
        img, label, landmark, bbox = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.landmarks[index]), copy.deepcopy(self.bboxs[index])
        ori_img_w, ori_img_h = img.size

        # Need AU Label
        if self.Experiment=='AU':
            if self.flag=='train':
                label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if label[i] < 0.5:
                        label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif label[i] < 1.0:
                        label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

        # Face Rotation
        # left  = np.min(landmark[:,0])
        # right = np.max(landmark[:,0])
        # upper = np.min(landmark[:,1])
        # lower = np.max(landmark[:,1])

        # BoundingBox
        left  = bbox[0]
        upper = bbox[1]
        right = bbox[2]
        lower = bbox[3]

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=upper

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        # Visualization
        # img_transform = img.resize((crop_img_w ,crop_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))

        if self.target_transform is not None:
            label = self.transform(label)

        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
        landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
        landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        if self.Model in ['ResNet-101', 'ResNet-50']:
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 11:
                        au_location[i,j] = 12 
                    if au_location[i,j]>=45: 
                        au_location[i,j] = 44
        elif self.Model == 'ResNet-18':
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 112)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 23:
                        au_location[i,j] = 24 
                    if au_location[i,j]>=89: 
                        au_location[i,j] = 88

        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        au_location = torch.LongTensor(au_location)

        if self.Experiment=='Fuse':
            if self.flag=='train':
                AU_label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if AU_label[i] < 0.5:
                        AU_label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif AU_label[i] < 1.0:
                        AU_label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        AU_label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                AU_label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

            return (trans_img, self.imgs[index]), au_location, (label, AU_label)
        
        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)

class MyDataset_EM_Artificial_noBbox(data.Dataset):
    def __init__(self, imgs, labels, landmarks, flag, needAU, Model, Experiment, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        self.Experiment = Experiment
        self.PriorKnowledgeTable_Train = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [1.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float)
        self.PriorKnowledgeTable_Test = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float)
        
    def __getitem__(self, index):
        img, label, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.landmarks[index])
        ori_img_w, ori_img_h = img.size

        # Need AU Label
        if self.Experiment=='AU':
            if self.flag=='train':
                label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if label[i] < 0.5:
                        label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif label[i] < 1.0:
                        label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

        # Face Rotation
        left  = np.min(landmark[:,0])
        right = np.max(landmark[:,0])
        upper = np.min(landmark[:,1])
        lower = np.max(landmark[:,1])

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.2)) # enlarge bbox, Defaults: 0.15
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=upper

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        # Visualization
        # img_transform = img.resize((crop_img_w ,crop_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))

        if self.target_transform is not None:
            label = self.transform(label)

        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
        landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
        landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        if self.Model in ['ResNet-101', 'ResNet-50']:
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 11:
                        au_location[i,j] = 12 
                    if au_location[i,j]>=45: 
                        au_location[i,j] = 44
        elif self.Model == 'ResNet-18':
            au_location = get_au_loc(copy.deepcopy(landmark),trans_img_w, 112)
            for i in range(au_location.shape[0]):
                for j in range(4):
                    if au_location[i,j]<= 23:
                        au_location[i,j] = 24 
                    if au_location[i,j]>=89: 
                        au_location[i,j] = 88

        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        au_location = torch.LongTensor(au_location)
        
        if self.Experiment=='Fuse':
            if self.flag=='train':
                AU_label = self.PriorKnowledgeTable_Train[label].reshape(self.PriorKnowledgeTable_Train.shape[1],)
                for i in range(self.PriorKnowledgeTable_Train.shape[1]):
                    if AU_label[i] < 0.5:
                        AU_label[i] = np.random.uniform(0.0,0.25) # plan1: (0.0,0.4), plan2: (0.0,0.2), plan3: (0.0,0.25)
                    elif AU_label[i] < 1.0:
                        AU_label[i] = np.random.uniform(0.5,0.75) # plan1: (0.4,0.7), plan2: (0.6,0.8), plan3: (0.5,0.75)
                    else:
                        AU_label[i] = np.random.uniform(0.75,1.0) # plan1: (0.7,1.0), plan2: (0.8,1.0), plan3: (0.75,1.0)
                # label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)
            elif self.flag=='test':
                AU_label = self.PriorKnowledgeTable_Test[label].reshape(self.PriorKnowledgeTable_Test.shape[1],)

            return (trans_img, self.imgs[index]), au_location, (label, AU_label)
        
        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)
