import torch
import torch.nn as nn

             
# Web Link : https://github.com/kapsdeep/FER/blob/master/prior_probability.ipynb 
class Expression_Independent_AU_Loss(nn.Module):
    def __init__(self, size_average=True):
        super(Expression_Independent_AU_Loss, self).__init__()

        self.size_average = size_average

        # self.positive_au_pairs = [(1,2), (4,7), (4,9), (7,9), (6,12), (9,17), (15,17), (15,24), (17,24), (23,24)]
        # self.negative_au_pairs = [(2,6), (2,7), (12,15), (12,17)]
        self.positive_au_pairs = [(0,1), (2,5), (2,6), (5,6), (4,8), (6,11), (9,11), (9,14), (11,14), (13,14)]
        self.negative_au_pairs = [(1,4), (1,5), (8,9), (8,11)]

    def forward(self, pred, target):

        positive_loss = torch.zeros(1).cuda()
        negative_loss = torch.zeros(1).cuda()

        # Positive Loss
        for i, j in self.positive_au_pairs:
            positive_loss += torch.clamp(self.get_pos_prob(pred, i) * self.get_pos_prob(pred, j) - self.get_pos_pos_prob(pred, i, j), min=0.0) + \
                             torch.clamp(self.get_neg_prob(pred, i) * self.get_pos_prob(pred, j) - self.get_pos_pos_prob(pred, i, j), min=0.0) + \
                             torch.clamp(self.get_pos_prob(pred, i) * self.get_neg_prob(pred, j) - self.get_pos_pos_prob(pred, i, j), min=0.0)

        # Negative Loss
        for i, j in self.negative_au_pairs:
            negative_loss += torch.clamp(self.get_pos_prob(pred, i) * self.get_pos_prob(pred, j) - self.get_pos_pos_prob(pred, i, j), min=0.0) + \
                             torch.clamp(self.get_pos_pos_prob(pred, i, j) - self.get_neg_prob(pred, i) * self.get_pos_prob(pred, j), min=0.0) + \
                             torch.clamp(self.get_pos_pos_prob(pred, i, j) - self.get_pos_prob(pred, i) * self.get_neg_prob(pred, j), min=0.0)

        # Batch Loss
        batch_loss = positive_loss + negative_loss

        if self.size_average:
            return batch_loss.mean()

        return batch_loss

    # Formula : P^{i_1}
    def get_pos_prob(self, pred, index):
        result = torch.zeros(1).cuda()
        for i in range(pred.size(0)):
            if pred[i, index] >= 0.5:
                result += pred[i, index]
        return result / pred.size(0)

    # Formula : P^{i_0}
    def get_neg_prob(self, pred, index):
        result = torch.zeros(1).cuda()
        for i in range(pred.size(0)):
            if pred[i, index] < 0.5:
                result += pred[i, index]
        return result / pred.size(0)

    # Formula : P^{(i_1)(j_0)}
    def get_pos_neg_prob(self, pred, index_1, index_2):
        result = torch.zeros(1).cuda()
        for i in range(pred.size(0)):
            if pred[i, index_1] >= 0.5 and pred[i, index_2] < 0.5:
                result += pred[i, index_1] * pred[i, index_2]
        return result / pred.size(0)

    # Formula : P^{(i_0)(j_1)}
    def get_neg_pos_prob(self, pred, index_1, index_2):
        result = torch.zeros(1).cuda()
        for i in range(pred.size(0)):
            if pred[i, index_1] < 0.5 and pred[i, index_2] >= 0.5:
                result += pred[i, index_1] * pred[i, index_2]
        return result / pred.size(0)

    # Formula : P^{(i_1)(j_1)}
    def get_pos_pos_prob(self, pred, index_1, index_2):
        result = torch.zeros(1).cuda()
        for i in range(pred.size(0)):
            if pred[i, index_1] >= 0.5 and pred[i, index_2] >= 0.5:
                result += pred[i, index_1] * pred[i, index_2]
        return result / pred.size(0)

class Generate_AU_Loss(nn.Module):
    def __init__(self, size_average=True):
        super(Generate_AU_Loss, self).__init__()

        self.size_average = size_average

        self.Mask_A = torch.Tensor([
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).float().cuda()

        self.Mask_B = torch.Tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).float().cuda()

        self.Mask_C = torch.Tensor([
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).float().cuda()

    def forward(self, target):
        batch_loss = self.Mask_A * torch.sqrt((target-(0.75 + 1.00)/2).pow(2)) + \
                     self.Mask_B * torch.sqrt((target-(0.50 + 0.75)/2).pow(2)) + \
                     self.Mask_C * torch.sqrt((target-(0.00 + 0.25)/2).pow(2))

        if self.size_average:
            return batch_loss.mean()
        
        return batch_loss.sum()

class MSELoss(nn.Module):
    def __init__(self, size_average=True):
        super(MSELoss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = (pred-target).pow(2)

        weight = target.clone()
        weight[weight >= 0.5] = 3
        weight[weight < 0.5] = 1

        loss = loss * weight

        if self.size_average:
            return loss.mean()

        return loss.sum()

class BCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(BCELoss, self).__init__()

        self.size_average = size_average

    def forward(self, pred, target):
        loss = - target * torch.log(pred)
        
        weight = target.clone()
        weight[weight >= 0.5] = 3
        weight[weight < 0.5] = 1

        loss = loss * weight

        if self.size_average:
            return loss.mean()

        return loss.sum()