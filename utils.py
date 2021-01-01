import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import get_au_loc, MyDataset_EM, MyDataset_AU, MyDataset_EM_Artificial, MyDataset_EM_Artificial_Compound, MyDataset_EM_Artificial_noBbox
from model import ResNet101, ResNet101_Compound

numOfAU = 17


class AverageMeter(object):
    '''Computes and stores the sum, count and average'''

    def __init__(self):
        self.reset()

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val 
        self.count += count
        if self.count==0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count


def str2bool(input):
    if isinstance(input, bool):
       return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def BulidDataloader(args, flag='train'):
    '''Bulid Datadloader'''

    # Set Transform
    print('Use 224 * 224 Image')
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
    target_trans = None

    # Set Data Path
    dataPath_prefix = 'your-data-path'
    ImagePath, BBoxPath, LandmarkPath, Train_List, Test_List = '', '', '', '', ''

    if args.Dataset == 'RAF':

        if args.Distribute == 'Basic':
            if args.Aligned:
                ImagePath = dataPath_prefix + '/RAF/basic/Image/aligned/'
                LandmarkPath = dataPath_prefix + '/RAF/basic/Annotation/Landmarks_68_FAN_aligned/'
            else:
                ImagePath = dataPath_prefix + '/RAF/basic/Image/original/'
                LandmarkPath = dataPath_prefix + '/RAF/basic/Annotation/Landmarks_68_FAN_bbox/'

            Train_List = dataPath_prefix + '/RAF/basic/EmoLabel/list_patition_label.txt'
            Test_List = dataPath_prefix + '/RAF/basic/EmoLabel/list_patition_label.txt'
            BBoxsPath = dataPath_prefix + '/RAF/basic/Annotation/boundingbox/'

        elif args.Distribute == 'Compound':
            if args.Aligned:
                ImagePath = dataPath_prefix + '/RAF/compound/Image/aligned/'
                LandmarkPath = dataPath_prefix + '/RAF/compound/Annotation/Landmarks_68_FAN_bbox/'
            else:
                ImagePath = dataPath_prefix + '/RAF/compound/Image/original/'
                LandmarkPath = dataPath_prefix + '/RAF/compound/Annotation/Landmarks_68_FAN_bbox/'

            Train_List = dataPath_prefix + '/RAF/compound/EmoLabel/list_patition_label.txt'
            Test_List = dataPath_prefix + '/RAF/compound/EmoLabel/list_patition_label.txt'
            BBoxsPath = dataPath_prefix + '/RAF/compound/Annotation/boundingbox/'

    elif args.Dataset=='BP4D':

        ImagePath = dataPath_prefix + '/BP4D/2D_img/'
        LandmarkPath = dataPath_prefix + '/BP4D/Annotations/Landmarks/'

        Train_List = dataPath_prefix + '/BP4D/list_experiment/id_fold1_2.txt'
        Test_List = dataPath_prefix + '/BP4D/list_experiment/id_fold3.txt'

    elif args.Dataset=='SFEW':

        ImagePath = dataPath_prefix + '/SFEW/'
        LandmarkPath = dataPath_prefix + '/SFEW/'

        Train_List = dataPath_prefix + '/SFEW/list_experiment/id_train_list.txt'
        Test_List = dataPath_prefix + '/SFEW/list_experiment/id_val_list.txt'

    elif args.Dataset=='MMI':

        ImagePath = dataPath_prefix + '/MMI/Select_Frames/Imgs_hyuan-cvpr18/'
        LandmarkPath = dataPath_prefix + '/MMI/Select_Frames/Landmarks_68_FAN/'

        Train_List = dataPath_prefix + '/MMI/Select_Frames/list/id_train_list_crossval0.txt'
        Test_List = dataPath_prefix + '/MMI/Select_Frames/list/id_val_list_crossval0.txt'

    elif args.Dataset=='ExpW':

        ImagePath = dataPath_prefix + '/ExpW/data/image/origin/'
        LandmarkPath = dataPath_prefix + '/ExpW/landmarks/ExpW_landmarks.txt'

        Train_List = dataPath_prefix + '/ExpW/data/label/label.lst'
        Test_List = dataPath_prefix + '/ExpW/data/label/label.lst'

    # Load Dataset
    data_imgs, data_labels, data_landmarks, data_bboxs = [], [], [], []

    if args.Dataset=='RAF':
        
        # Basic Notes: { 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral}
        # Compound Notes: { 1: Happily Surprised, 2: Happily Disgusted, 3: Sadly Fearful, 4: Sadly Angry, 5: Sadly Surprised, 6: Sadly Disgusted, 7: Fearfully Angry, 8: Fearfully Surprised, 9: Angrily Surprised, 10: Angrily Disgusted, 11: Disgustedly Surprised}

        if flag == 'train':
            
            list_patition_label = pd.read_csv(Train_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)

            for i in range(list_patition_label.shape[0]):
                if list_patition_label[i, 0][:5] == "train":

                    if not os.path.exists(LandmarkPath+list_patition_label[i, 0][:-4]+'.txt'):
                        continue
                    landmark = np.loadtxt(LandmarkPath+list_patition_label[i, 0][:-4]+'.txt')
                    if landmark.ndim < 2:
                        continue

                    bbox = np.loadtxt(BBoxsPath+list_patition_label[i, 0][:-4]+'.txt')
                    landmark[:, 0] += bbox[0]
                    landmark[:, 1] += bbox[1]
                    
                    data_imgs.append(ImagePath+list_patition_label[i, 0])
                    data_labels.append(list_patition_label[i, 1]-1)
                    data_landmarks.append(landmark)
                    data_bboxs.append(bbox)

        elif flag == 'test':

            list_patition_label = pd.read_csv(Test_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)

            for i in range(list_patition_label.shape[0]):
                if list_patition_label[i, 0][:4] == "test":

                    if not os.path.exists(LandmarkPath+list_patition_label[i, 0][:-4]+'.txt'):
                        continue
                    landmark = np.loadtxt(LandmarkPath+list_patition_label[i, 0][:-4]+'.txt')
                    if landmark.ndim < 2:
                        continue

                    bbox = np.loadtxt(BBoxsPath+list_patition_label[i, 0][:-4]+'.txt')
                    landmark[:, 0] += bbox[0]
                    landmark[:, 1] += bbox[1]

                    data_imgs.append(ImagePath + list_patition_label[i, 0])
                    data_labels.append(list_patition_label[i, 1]-1)
                    data_landmarks.append(landmark)
                    data_bboxs.append(bbox)

        # Dataset Distribute
        if flag == 'train':
            print('The train dataset distribute: %d, %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks), len(data_bboxs)) )
        elif flag == 'test':
            print('The test dataset distribute: %d, %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks), len(data_bboxs)) )

        # Dataset
        needAU = False if args.Experiment == 'EM' else True                    
        dataset = MyDataset_EM(data_imgs, data_labels, data_landmarks, data_bboxs, flag, needAU, args.Model, trans, target_trans)

        # DataLoader
        if flag=='train':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
        elif flag=='test':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Test_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)

        return data_loader

    elif args.Dataset=='BP4D':
    
        List_Path = Train_List if flag == 'train' else Test_List
        
        numOfData = 0
        
        with open(List_Path,'r') as f:
            lines = f.readlines()
            
            for line in lines:
                id = line[:-1]
                if not (os.path.exists(ImagePath + id + '.jpg') and os.path.exists(dataPath_prefix + '/BP4D/Annotations/AUs/' + id + '.txt') and os.path.exists(LandmarkPath + id + '.txt')):
                    continue

                landmark = np.loadtxt(LandmarkPath + id + '.txt')
                if landmark.ndim < 2:
                    continue

                numOfData+=1
                if numOfData%10000==0:
                    print('Load Data Num: %d' % numOfData)

                # load img path
                data_imgs.append(ImagePath+id+'.jpg')

                # load label
                label_txt = np.loadtxt(dataPath_prefix + '/BP4D/Annotations/AUs/'+ id + '.txt')
                label = np.zeros(12, dtype=np.float32)

                if label_txt.size == 0:
                    data_labels.append(label)
                else:
                    if label_txt.ndim == 1: # only one au
                        label_txt = label_txt[np.newaxis, :]

                    for au in label_txt:  # au: [au_idx, au_value]
                        au_idx = au[0]
                        label[int(au_idx)] = 1

                    data_labels.append(label)

                # load landmark 
                data_landmarks.append(landmark)

        # Dataset Distribute
        if flag == 'train':
            print('The train dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))
        elif flag == 'test':
            print('The test dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))

        # Dataset with Bbox
        dataset = MyDataset_AU(data_imgs, data_labels, data_landmarks, flag, args.Model, trans, target_trans)

        # DataLoader
        if flag=='train':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
        elif flag=='test':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Test_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)

        return data_loader

    elif args.Dataset=='SFEW':
        
        # Basic Notes: {1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral}
        Label = {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happy':3, 'Sad':4, 'Angry':5, 'Neutral':6}

        if flag == 'train':
            list_patition_label = pd.read_csv(Train_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            list_patition_label = list_patition_label.reshape(list_patition_label.shape[0],)

            for i in range(list_patition_label.shape[0]):
                if not os.path.exists(LandmarkPath + 'Train/Annotations/Landmarks_68_FAN/' + list_patition_label[i] + '.txt'):
                    continue

                landmark = np.loadtxt(LandmarkPath + 'Train/Annotations/Landmarks_68_FAN/' + list_patition_label[i] + '.txt')
                if landmark.ndim < 2:
                    continue
                
                if os.path.exists(ImagePath + 'Train/imgs/' + list_patition_label[i] + '.jpg'):
                    data_imgs.append(ImagePath + 'Train/imgs/' + list_patition_label[i] + '.jpg')

                elif os.path.exists(ImagePath + 'Train/imgs/' + list_patition_label[i] + '.png'):
                    data_imgs.append(ImagePath + 'Train/imgs/' + list_patition_label[i] + '.png')

                data_labels.append(Label[list_patition_label[i].split('/',1)[0]])
                data_landmarks.append(landmark)

        elif flag == 'test':

            list_patition_label = pd.read_csv(Test_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            list_patition_label = list_patition_label.reshape(list_patition_label.shape[0],)

            for i in range(list_patition_label.shape[0]):
                if not os.path.exists(LandmarkPath + 'Val/Annotations/Landmarks_68_FAN/' + list_patition_label[i] + '.txt'):
                    continue
                landmark = np.loadtxt(LandmarkPath + 'Val/Annotations/Landmarks_68_FAN/' + list_patition_label[i] + '.txt')
                if landmark.ndim<2:
                    continue

                if os.path.exists(ImagePath + 'Val/imgs/' + list_patition_label[i] + '.jpg'):
                    data_imgs.append(ImagePath + 'Val/imgs/' + list_patition_label[i] + '.jpg')
                elif os.path.exists(ImagePath + 'Val/imgs/' + list_patition_label[i] + '.png'):
                    data_imgs.append(ImagePath + 'Val/imgs/' + list_patition_label[i] + '.png')

                data_labels.append(Label[list_patition_label[i].split('/',1)[0]])
                data_landmarks.append(landmark)

        # Dataset Distribute
        if flag == 'train':
            print('The train dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))
        elif flag == 'test':
            print('The test dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))

        # Dataset
        needAU = False if args.Experiment == 'EM' else True                    
        dataset = MyDataset_EM_Artificial_noBbox(data_imgs, data_labels, data_landmarks, flag, needAU, args.Model, args.Experiment, trans, target_trans)

        # DataLoader
        if flag == 'train':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
        elif flag == 'test':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Test_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)

        return data_loader

    elif args.Dataset=='MMI':
        
        # Basic Notes: {1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral}
        Label = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5}

        if flag == 'train':

            list_patition_label = pd.read_csv(Train_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            list_patition_label = list_patition_label.reshape(list_patition_label.shape[0],)

            for i in range(list_patition_label.shape[0]):
                if not os.path.exists(LandmarkPath + list_patition_label[i] + '.txt'):
                    continue
                landmark = np.loadtxt(LandmarkPath + list_patition_label[i] + '.txt')
                if landmark.ndim<2:
                    continue

                data_imgs.append(ImagePath + list_patition_label[i] + '.jpg')
                label = pd.read_csv(dataPath_prefix + '/MMI/Select_Frames/Emotions_hyuan-cvpr18/' + list_patition_label[i] + '.txt', header=None, delim_whitespace=True)
                label = np.array(label)[0,0]
                data_labels.append(Label[label])
                data_landmarks.append(landmark)

        elif flag == 'test':

            list_patition_label = pd.read_csv(Test_List, header=None, delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            list_patition_label = list_patition_label.reshape(list_patition_label.shape[0],)

            for i in range(list_patition_label.shape[0]):
                if not os.path.exists(LandmarkPath + list_patition_label[i] + '.txt'):
                    continue
                landmark = np.loadtxt(LandmarkPath + list_patition_label[i] + '.txt')
                if landmark.ndim<2:
                    continue

                data_imgs.append(ImagePath + list_patition_label[i] + '.jpg')
                label = pd.read_csv(dataPath_prefix + '/MMI/Select_Frames/Emotions_hyuan-cvpr18/' + list_patition_label[i] + '.txt', header=None, delim_whitespace=True)
                label = np.array(label)[0,0]
                data_labels.append(Label[label])
                data_landmarks.append(landmark)

        # Dataset Distribute
        if flag == 'train':
            print('The train dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))
        elif flag == 'test':
            print('The test dataset distribute: %d, %d, %d' % (len(data_imgs), len(data_labels), len(data_landmarks)))

        # Dataset
        needAU = False if args.Experiment == 'EM' else True                    
        dataset = MyDataset_EM_Artificial_noBbox(data_imgs, data_labels, data_landmarks, flag, needAU, args.Model, args.Experiment, trans, target_trans)

        # DataLoader
        if flag == 'train':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
        elif flag ==' test':
            data_loader = data.DataLoader(dataset=dataset, batch_size=args.Test_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)
        
        return data_loader
    
    elif args.Dataset=='ExpW':
        
        # Basic Notes: {1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral}
        Label = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }

        list_patition_label = pd.read_csv(dataPath_prefix + '/ExpW/landmarks/ExpW_landmarks.txt', header=None, delim_whitespace=True)
        list_patition_label = np.array(list_patition_label)

        for i in range(list_patition_label.shape[0]):

            landmark = list_patition_label[i,8:].reshape(68,2).astype(np.int)

            bbox = list_patition_label[i,2:6].astype(np.int)
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[2], bbox[3]
            
            landmark[:,0]+=bbox[0]
            landmark[:,1]+=bbox[1]

            data_imgs.append(args.ImagePath+list_patition_label[i,0])
            label = list_patition_label[i,7]
            data_labels.append(Label[label])
            data_landmarks.append(landmark)
            data_bboxs.append(bbox)

        # Dataset Distribute
        print('The dataset distribute: %d, %d, %d, %d' % ( len(data_imgs), len(data_labels), len(data_landmarks), len(data_bboxs) ) )

        # Dataset
        needAU = False if args.Experiment == 'EM' else True                    
        dataset = MyDataset_EM_Artificial(data_imgs, data_labels, data_landmarks, data_bboxs, flag, needAU, args.Model, args.Experiment, trans, target_trans)
        
        torch.manual_seed(1)              # Set CPU Seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)     # Set Current GPU Seed
            torch.cuda.manual_seed_all(1) # Set All GPU Seed

        trainSet_size = int(0.9 * len(data_imgs))
        testSet_size = len(data_imgs) - trainSet_size
        train_set, test_set = data.random_split( dataset, [trainSet_size, testSet_size] )

        print('The num of TrainSet and TestSet: %d , %d ' % (trainSet_size, testSet_size) )

        # DataLoader
        train_loader = data.DataLoader(dataset=train_set, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=True)
        test_loader = data.DataLoader(dataset=test_set, batch_size=args.Train_Batch_Size, shuffle=True, num_workers=args.Num_Workers, drop_last=False)

        return train_loader, test_loader


def Bulid_Model(args):
    '''Bulid Model'''

    if args.Distribute == 'Basic':
        model = ResNet101(args.Dim)

    elif args.Distribute == 'Compound':
        model = ResNet101_Compound(args.Dim)


    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)

        # Save GPU Memory
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print('No Resume Model')

    if args.DataParallel:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def Set_Param_Optim(args, model):
    '''Set parameters optimizer'''

    # Expression Recognition Experiment
    if args.Experiment == 'EM': 
        for param in model.parameters():
            param.requires_grad = False

        for param in model.backbone.parameters():
            param.requires_grad = True
        for param in model.LRN_em.parameters():
            param.requires_grad = True
        for param in model.reduce_dim_em.parameters():
            param.requires_grad = True
        for param in model.pred_em.parameters():
            param.requires_grad = True 

        param_optim = filter(lambda p:p.requires_grad, model.parameters())

    # AU Recognition Experiment
    elif args.Experiment == 'AU':
        for param in model.parameters():
            param.requires_grad = False

        for param in model.deconv_layer1.parameters():
            param.requires_grad = True
        for param in model.deconv_layer2.parameters():
            param.requires_grad = True
        for param in model.deconv_layer3.parameters():
            param.requires_grad = True

        for param in model.reduce_dim_1_au.parameters():
            param.requires_grad = True
        for param in model.reduce_dim_2_au.parameters():
            param.requires_grad = True
        for param in model.reduce_dim_3_au.parameters():
            param.requires_grad = True
        
        for param in model.LRN_au.parameters():
            param.requires_grad = True
        for param in model.Crop_Net_1.parameters():
            param.requires_grad = True
        for param in model.Crop_Net_2.parameters():
            param.requires_grad = True
        for param in model.pred_au.parameters():
            param.requires_grad = True

        model.PriorKnowledgeTable.requires_grad = True

        param_optim = filter(lambda p:p.requires_grad, model.parameters())

    # Feature Fuse Experiment
    elif args.Experiment == 'Fuse':
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc_em_fuse_3.parameters():
            param.requires_grad = True
        for param in model.fc_au_fuse_3.parameters():
            param.requires_grad = True

        for param in model.fc_attention_fuse_3.parameters():
            param.requires_grad = True

        for param in model.pred_em_fuse_3.parameters():
            param.requires_grad = True
        for param in model.pred_em.parameters():
            param.requires_grad = True

        param_optim = filter(lambda p:p.requires_grad, model.parameters())
    
    return param_optim


def Set_Criterion_Optimizer(args, param_optim):
    '''Set Criterion and Optimizer'''

    optimizer = optim.SGD(param_optim, lr=args.LearnRate, momentum=0.9)

    return optimizer


def Adjust_Learning_Rate(optimizer, epoch, LR):
    '''Adjust Learning Rate'''
    # lr = 0.001

    if epoch<=15:
        lr = LR
    elif epoch<=30:
        lr = 0.1 * LR
    else:
        lr = 0.01 * LR

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples and the loss'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    if args.Distribute == 'Basic':
        numOfLabel = 7
    elif args.Distribute == 'Compound':
        numOfLabel = 11
     
    for index in range(numOfLabel):
        TP = np.sum((pred == index) * (target == index))
        TN = np.sum((pred != index) * (target != index))

        # Compute Accuracy of All --> TP+TN / All
        acc_1[index].update(np.sum(pred == target), pred.shape[0])
        acc_2[index].update(TP, np.sum(target == index))

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[index].update(TP, np.sum(pred == index))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[index].update(TP, np.sum(target == index))

    # Compute Loss
    loss.update(float(loss_.cpu().data.numpy()))


def Compute_Accuracy_AU(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples and the loss'''

    pred = pred.cpu().data.numpy()
    pred[pred < 0.5] = 0
    pred[pred > 0] = 1

    target = target.cpu().data.numpy()
    target[target < 0.5] = 0
    target[target > 0] = 1

    pred = pred.astype(np.int32).reshape(pred.shape[0], pred.shape[1])
    target = target.astype(np.int32).reshape(target.shape[0], target.shape[1])

    for index in range(numOfAU):
        TP = np.sum((pred[:, index] == 1) * (target[:, index] == 1))
        TN = np.sum((pred[:, index] == 0) * (target[:, index] == 0))

        # Compute Accuracy of All --> TP+TN / All
        acc_1[index].update(TP + TN, pred.shape[0])
        acc_2[index].update(TP + TN, pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[index].update(TP, np.sum(pred[:, index] == 1))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[index].update(TP, np.sum(target[:, index]==1))

    # Compute Loss
    loss.update(float(loss_.cpu().data.numpy()))


def Show_Accuracy(acc_1, acc_2, prec, recall, numOfClass=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # compute F1 value    
    f1 = [AverageMeter() for i in range(numOfClass)]
    for i in range(numOfClass):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2 * prec[i].avg * recall[i].avg / (prec[i].avg + recall[i].avg)
    
    acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0, 0
    for i in range(numOfClass):
        acc_1_avg += acc_1[i].avg
        acc_2_avg += acc_2[i].avg
        prec_avg += prec[i].avg
        recall_avg += recall[i].avg
        f1_avg += f1[i].avg
    acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = acc_1_avg/numOfClass, acc_2_avg/numOfClass, prec_avg/numOfClass, recall_avg/numOfClass, f1_avg/numOfClass

    # Log Accuracy Infomation
    Accuracy_Info = ''
    
    Accuracy_Info+='    Accuracy(1)'
    for i in range(numOfClass):
        Accuracy_Info+=' {:.4f}'.format(acc_1[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Accuracy(2)'
    for i in range(numOfClass):
        Accuracy_Info+=' {:.4f}'.format(acc_2[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Precision'
    for i in range(numOfClass):
        Accuracy_Info+=' {:.4f}'.format(prec[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Recall'
    for i in range(numOfClass):
        Accuracy_Info+=' {:.4f}'.format(recall[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    F1'
    for i in range(numOfClass):
        Accuracy_Info+=' {:.4f}'.format(f1[i].avg)
    Accuracy_Info+='\n'

    return Accuracy_Info, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg