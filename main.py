import time
import warnings


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import *
from loss import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Facial Expression Recognition Experiment')

parser.add_argument('--Log_Name', type=str, help='Naming Format: date_experiment_model')
parser.add_argument('--Experiment', default='EM', type=str, choices=['EM', 'AU', 'Fuse'], 
                    help='1->Expression Recognition Experiment, 2->AU Recognition Experiment, 3->Feature Fuse Experiment')

parser.add_argument('--Dataset', default='RAF', type=str, choices=['RAF', 'SFEW', 'MMI', 'ExpW', 'BP4D'], help='Value Range: RAF, BP4D, SFEW, MMI, ExpW')
parser.add_argument('--Distribute', default='Basic', type=str, choices=['Basic', 'Compound'], help='Value Range: Basic, Compound')
parser.add_argument('--Aligned', default=False, type=str2bool, help='whether to Aligned Image')

parser.add_argument('--Model', default='ResNet-101', type=str, choices=['ResNet-101', 'ResNet-50', 'ResNet-18'],
                    help='1->ResNet-101(pre-trained on ImageNet), 2->ResNet-50(pre-trained on ImageNet), 3->ResNet-18(pre-trained on ImageNet)')
parser.add_argument('--Resume_Model', default='None', type=str, help='if Resume_Model == none, then load pre-trained on ImageNet from PyTorch')

parser.add_argument('--Dim', default=1024, type=int, help='Dim Of Fuse Feature')
parser.add_argument('--numOfAU', default=17, type=int, help='Number of Action Units')
parser.add_argument('--numOfLabel', default=7, type=int, help='Number of Expression Labels')

parser.add_argument('--Epoch', default=40, type=int, help='Epoch')
parser.add_argument('--LearnRate', default=0.01, type=float, help='Learning Rate')
parser.add_argument('--Train_Batch_Size', default=64, type=int, help='Batch Size during training')
parser.add_argument('--Test_Batch_Size', default=64, type=int, help='Batch Size during testing')

parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--Num_Workers', default=12, type=int, help='Number of Workers')
parser.add_argument('--DataParallel', default=False, type=str2bool, help='Data Parallel')

def Train(args, model, criterion, optimizer, train_loader, writer, epoch):

    numOfClass = args.numOfAU if args.Experiment == 'AU' else args.numOfLabel
    acc_1, acc_2, prec, recall = [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    model.train()
    if args.Experiment == 'Fuse':
        model.backbone.eval()

    for i in range(numOfClass):
        acc_1[i].reset()
        acc_2[i].reset()
        prec[i].reset()
        recall[i].reset()

    loss.reset()
    data_time.reset()
    batch_time.reset()

    optimizer, lr = Adjust_Learning_Rate(optimizer, epoch, args.LearnRate)

    end = time.time()
    for step, (input, au_loc, target) in enumerate(train_loader, start=1):

        input, imgPath = input
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time()-end)

        if args.Experiment in ['AU', 'Fuse']:
            au_loc = au_loc.cuda()
            au_target = model.get_au_target(target.cpu()) # generate au label

        # forward
        if args.Experiment == 'EM':
            pred = model(input, args)
            loss_ = criterion(pred, target)

        elif args.Experiment == 'AU':
            pred = model((input, au_loc), args)
            loss_ = criterion(pred, au_target) + 0.5 * Expression_Independent_AU_Loss()(pred, au_target) + Generate_AU_Loss()(model.PriorKnowledgeTable)

        elif args.Experiment == 'Fuse':
            pred1, pred2, au_prob = model((input, au_loc), args)
            loss_ = criterion(pred1, target) + criterion(pred2, target) + 0.05 * Expression_Independent_AU_Loss()(au_prob, au_target)

        # backward
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step() 

        # compute accuracy, recall and loss
        if args.Experiment == 'EM':
            Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss)

        elif args.Experiment == 'AU':
            Compute_Accuracy_AU(args, pred, au_target, loss_, acc_1, acc_2, prec, recall, loss)

        elif args.Experiment == 'Fuse':
            Compute_Accuracy_Expression(args, pred2, target, loss_, acc_1, acc_2, prec, recall, loss)

        batch_time.update(time.time()-end)
        end = time.time()

    Accuracy_Info, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_1, acc_2, prec, recall, numOfClass=numOfClass)

    # writer
    writer.add_scalar('Accuracy_1', acc_1_avg, epoch)
    writer.add_scalar('Accuracy_2', acc_2_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)
    writer.add_scalar('Loss', loss.avg, epoch)

    LogInfo = '''
    [Tain ({exp})]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, lr, data_time=data_time, batch_time=batch_time, exp=args.Experiment)
            
    LogInfo += Accuracy_Info

    LogInfo += '''    Acc_avg(1) {0:.4f} Acc_avg(2) {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Loss {loss.avg:.4f}'''.format(acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss=loss)

    print(LogInfo)         
   
def Test(args, model, criterion, optimizer, test_loader, writer, epoch, Best_Accuracy):

    numOfClass = args.numOfAU if args.Experiment == 'AU' else args.numOfLabel

    acc_1, acc_2, prec, recall = [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in range(numOfClass)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    # Test Model
    model.eval()

    for i in range(numOfClass):
        acc_1[i].reset()
        acc_2[i].reset()
        prec[i].reset()
        recall[i].reset()

    loss.reset()
    data_time.reset()
    batch_time.reset()

    end = time.time()
    for step, (input, au_loc, target) in enumerate(test_loader, start=1):

        input, imgPath = input
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time()-end)

        if args.Experiment in ['AU', 'Fuse']:
            au_loc = au_loc.cuda()
            au_target = model.get_au_target(target.cpu()) # generate au label

        with torch.no_grad():

            # forward
            if args.Experiment == 'EM':
                pred = model(input, args)
                loss_ = criterion(pred, target)

            elif args.Experiment == 'AU':
                pred = model((input, au_loc), args)
                loss_ = criterion(pred, au_target) + 0.5 * Expression_Independent_AU_Loss()(pred, au_target) + Generate_AU_Loss()(model.PriorKnowledgeTable)

            elif args.Experiment == 'Fuse':
                pred1, pred2, au_prob = model((input, au_loc), args)
                loss_ = criterion(pred1, target) + criterion(pred2, target) + 0.05 * Expression_Independent_AU_Loss()(au_prob, au_target)

        # compute accuracy, recall and loss
        if args.Experiment == 'EM':
            Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss)

        elif args.Experiment == 'AU':
            Compute_Accuracy_AU(args, pred, au_target, loss_, acc_1, acc_2, prec, recall, loss)

        elif args.Experiment == 'Fuse':
            Compute_Accuracy_Expression(args, pred2, target, loss_, acc_1, acc_2, prec, recall, loss)

        batch_time.update(time.time()-end)
        end = time.time()

    Accuracy_Info, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_1, acc_2, prec, recall, numOfClass=numOfClass)

    # writer
    writer.add_scalar('Accuracy_1', acc_1_avg, epoch)
    writer.add_scalar('Accuracy_2', acc_2_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)
    writer.add_scalar('Loss', loss.avg, epoch)

    LogInfo = '''
    [Test ({exp})]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(epoch, data_time=data_time, batch_time=batch_time, exp=args.Experiment)
            
    LogInfo += Accuracy_Info

    LogInfo += '''    Acc_avg(1) {0:.4f} Acc_avg(2) {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Loss {loss.avg:.4f}'''.format(acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss=loss)

    print(LogInfo)

    # Save Checkpoints
    if acc_2_avg > Best_Accuracy:
        
        Best_Accuracy, Best_Epoch = acc_2_avg, epoch
        print('[Save] Best Acc: %.4f, Best Epoch: %d' % (Best_Accuracy, Best_Epoch))

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), '{}.pkl'.format(args.Log_Name))
        else:
            torch.save(model.state_dict(), '{}.pkl'.format(args.Log_Name))

    return Best_Accuracy

def main():
    '''main'''

    # Parse Argument
    args = parser.parse_args()

    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Backbone: %s' % args.Model)
    print('Experiment: %s' % args.Experiment)
    print('Resume_Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Dataset: %s' % args.Dataset)
    print('Distribute: %s' % args.Distribute)
    print('Use Aligned Image' if args.Aligned else 'Don\'t use Aligned Image')

    print('================================================')

    if args.Distribute == 'Basic':
        args.numOfLabel = 7
    elif args.Distribute == 'Compound':
        args.numOfLabel = 11

    print('Dim: %d' % args.Dim)
    print('Number Of Action Units: %d' % args.numOfAU)
    print('Number Of Expression Labels: %d' % args.numOfLabel)

    print('================================================')

    print('Number of Workers: %d' % args.Num_Workers)
    print('Use Data Parallel' if args.DataParallel else 'Dont\'t use Data Parallel')
    print('Epoch: %d' % args.Epoch)
    print('Train Batch Size: %d' % args.Train_Batch_Size)
    print('Test Batch Size: %d' % args.Test_Batch_Size)

    print('================================================')

    # Bulid Model
    print('Load Model...')
    model = Bulid_Model(args)
    print('Done!')

    print('================================================')
    
    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Criterion_Optimizer(args, param_optim)
    print('Done!')

    print('================================================')

    # Bulid Dataloader
    print("Buliding Train and Test Dataloader...")
    if args.Dataset == 'ExpW':
        train_loader, test_loader = BulidDataloader(args)
    else:
        train_loader = BulidDataloader(args, flag='train')
        test_loader = BulidDataloader(args, flag='test')
    print('Done!')

    print('================================================')

    Best_Accuracy = 0
    
    if args.Experiment in ['EM', 'Fuse']:
        criterion = nn.CrossEntropyLoss()
    elif args.Experiment == 'AU':
        criterion = MSELoss()

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter('{}'.format(args.Log_Name))
    
    for epoch in range(1, args.Epoch + 1):

        Train(args, model, criterion, optimizer, train_loader, writer, epoch)
        Best_Accuracy = Test(args, model, criterion, optimizer, test_loader, writer, epoch, Best_Accuracy)

        torch.cuda.empty_cache()

    writer.close()

if __name__=='__main__':
    main()
