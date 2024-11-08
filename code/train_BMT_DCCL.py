import argparse
import logging
import os
import random
import shutil
import sys


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, feature_memory, contrastive_losses
from val_2D import test_single_volume_contr, test_single_volume_edge_contr

from bezier_curve import bezier_curve
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Pseudo_Supervision', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UNet_contr', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2023, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
# CPS model2
parser.add_argument('--model2', type=str,
                    default='UNet_BMT_DCCL', help='model_name')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')


#  cutmix
parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
args = parser.parse_args()

def nonlinear_transformation(slices, flag = True):

    if flag:
        random_num = random.random()
        if random_num <= 0.4:
            return (slices + 1) / 2
        if random_num > 0.4 and random_num <= 0.7:
            points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
            xvals_2, yvals_2 = bezier_curve(points_2, nTimes=10000)
            xvals_2 = np.sort(xvals_2)
            yvals_2 = np.sort(yvals_2)
            nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)
            return (nonlinear_slices_2 + 1) / 2
        if random_num > 0.7:
            points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
            xvals_4, yvals_4 = bezier_curve(points_4, nTimes=10000)
            xvals_4 = np.sort(xvals_4)
            yvals_4 = np.sort(yvals_4)
            nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)
            return (nonlinear_slices_4 + 1) / 2
    else:
        random_num = random.random()
        if random_num <= 0.4:
            points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
            xvals_1, yvals_1 = bezier_curve(points_1, nTimes=10000)
            xvals_1 = np.sort(xvals_1)
            nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
            nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
            return (nonlinear_slices_1 + 1) / 2
        if random_num > 0.4 and random_num <= 0.7:
            points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
            xvals_3, yvals_3 = bezier_curve(points_3, nTimes=10000)
            xvals_3 = np.sort(xvals_3)
            nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
            nonlinear_slices_3[nonlinear_slices_3 == 1] = -1
            return (nonlinear_slices_3 + 1) / 2
        if random_num > 0.7:
            points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
            xvals_5, yvals_5 = bezier_curve(points_5, nTimes=10000)
            xvals_5 = np.sort(xvals_5)
            nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
            nonlinear_slices_5[nonlinear_slices_5 == 1] = -1
            return (nonlinear_slices_5 + 1) / 2

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
Good_student = 0 # 0: model1  1:model2

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = net_factory(net_type=args.model2, in_chns=1,
                            class_num=num_classes)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    edge_loss = torch.nn.BCEWithLogitsLoss()


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            volume_batch_model1 = torch.zeros_like(volume_batch)
            volume_batch_model2 = torch.zeros_like(volume_batch)
            for i in range(volume_batch.shape[0]):
                    slices = nonlinear_transformation(volume_batch[i, 0].cpu().numpy() * 2 - 1, True)
                    volume_batch_model1[i, 0] = torch.from_numpy(slices).cuda()
            for i in range(volume_batch.shape[0]):
                    slices = nonlinear_transformation(volume_batch[i, 0].cpu().numpy() * 2 - 1, True)
                    volume_batch_model2[i, 0] = torch.from_numpy(slices).cuda()

            outputs1, embedding1 = model1(volume_batch_model1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2, edge_combined_prediction, edge_predictions_list, embedding2 = model2(volume_batch_model2)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            dice_loss_1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            dice_loss_2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
             # contrastive
            labeled_features1 = embedding1[:args.labeled_bs,...]
            labeled_features2 = embedding2[:args.labeled_bs,...]
            unlabeled_features1 = embedding1[args.labeled_bs:,...]
            unlabeled_features2 = embedding2[args.labeled_bs:,...]
            true_labels = label_batch[:args.labeled_bs]
            
            if dice_loss_1 < dice_loss_2:
                Good_student = 0
            else:
                Good_student = 1

            if Good_student == 0:
                y = outputs_soft1[:args.labeled_bs]
                labeled_features = labeled_features1.permute(0, 2, 3, 1)
                _, prediction_label = torch.max(y, dim=1)
                 # select the correct predictions and ignore the background class
                mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
                 # Apply the filter mask to the features and its labels          
                labels_correct = true_labels[mask_prediction_correctly]
                labeled_features_correct = labeled_features[mask_prediction_correctly, ...]
                 # get projected features
                with torch.no_grad():
                    model1.eval()
                    proj_labeled_features_correct = model1.projection_head(labeled_features_correct)
                    model1.train()
                # updated memory bank           
                prototype_memory.add_features_from_sample_learned(model1, proj_labeled_features_correct, labels_correct)
                
            elif Good_student == 1:
                y = outputs_soft2[:args.labeled_bs]
                labeled_features = labeled_features2.permute(0, 2, 3, 1)
                _, prediction_label = torch.max(y, dim=1)
                 # select the correct predictions and ignore the background class
                mask_prediction_correctly = ((prediction_label == true_labels).float() * (prediction_label > 0).float()).bool()
                 # Apply the filter mask to the features and its labels          
                labels_correct = true_labels[mask_prediction_correctly]
                labeled_features_correct = labeled_features[mask_prediction_correctly, ...]
                 # get projected features
                with torch.no_grad():
                    model2.eval()
                    proj_labeled_features_correct = model2.projection_head(labeled_features_correct)
                    model2.train()
                # updated memory bank           
                prototype_memory.add_features_from_sample_learned(model2, proj_labeled_features_correct, labels_correct)
                
            
            _, pseudo_label_1 = torch.max(outputs_soft1[args.labeled_bs:], dim=1)  # Get pseudolabels
            _, pseudo_label_2 = torch.max(outputs_soft2[args.labeled_bs:], dim=1) 
                  
            
            # contrastive loss for model2
            labeled_features2 = labeled_features2.permute(0, 2, 3, 1)
            labeled_features_all_2 = labeled_features2.reshape(-1, labeled_features2.size()[-1])
            labeled_labels = true_labels.reshape(-1)
            # get predicted features
            proj_labeled_features_all_2 = model2.projection_head(labeled_features_all_2)
            pred_labeled_features_all_2 = model2.prediction_head(proj_labeled_features_all_2)

            # Apply contrastive learning loss
            loss_contr_labeled2 = contrastive_losses.contrastive_class_to_class_learned_memory(model2, pred_labeled_features_all_2, labeled_labels, num_classes, prototype_memory.memory)

            unlabeled_features2 = unlabeled_features2.permute(0, 2, 3, 1)
            unlabeled_features2 = unlabeled_features2.reshape(-1, labeled_features2.size()[-1])
            pseudo_label2 = pseudo_label_2.reshape(-1)

            # get predicted features
            proj_feat_unlabeled2 = model2.projection_head(unlabeled_features2)
            pred_feat_unlabeled2 = model2.prediction_head(proj_feat_unlabeled2)

            # Apply contrastive learning loss
            loss_contr_unlabeled2 = contrastive_losses.contrastive_class_to_class_learned_memory(model2, pred_feat_unlabeled2, pseudo_label2, num_classes, prototype_memory.memory)

            # contrastive loss for model1
            labeled_features1 = labeled_features1.permute(0, 2, 3, 1)
            labeled_features_all_1 = labeled_features1.reshape(-1, labeled_features1.size()[-1])
            labeled_labels = true_labels.reshape(-1)

            # get predicted features
            proj_labeled_features_all_1 = model1.projection_head(labeled_features_all_1)
            pred_labeled_features_all_1 = model1.prediction_head(proj_labeled_features_all_1)

            # Apply contrastive learning loss
            loss_contr_labeled_1 = contrastive_losses.contrastive_class_to_class_learned_memory(model1, pred_labeled_features_all_1, labeled_labels, num_classes, prototype_memory.memory)

            unlabeled_features1 = unlabeled_features1.permute(0, 2, 3, 1)
            unlabeled_features_1 = unlabeled_features1.reshape(-1, unlabeled_features1.size()[-1])
            pseudo_label1 = pseudo_label_1.reshape(-1)

            # get predicted features
            proj_feat_unlabeled_1 = model1.projection_head(unlabeled_features_1)
            pred_feat_unlabeled_1 = model1.prediction_head(proj_feat_unlabeled_1)

            # Apply contrastive learning loss
            loss_contr_unlabeled_1 = contrastive_losses.contrastive_class_to_class_learned_memory(model1, pred_feat_unlabeled_1, pseudo_label1, num_classes, prototype_memory.memory)

            loss1_sup = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss_1)
            loss2_sup = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss_2)

            
            # cutmix
            unlabeled_batch_1 = volume_batch_model1[args.labeled_bs:,...].clone()
            unlabeled_batch_2 = volume_batch_model2[args.labeled_bs:,...].clone()
            

            outputs1_unlabeled = outputs1[args.labeled_bs:,...].detach()
            outputs2_unlabeled = outputs2[args.labeled_bs:,...].detach()
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(unlabeled_batch_1.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(unlabeled_batch_1.size(), lam)
                
                unlabeled_batch_1[:, :, bbx1:bbx2, bby1:bby2] = unlabeled_batch_1[rand_index, :, bbx1:bbx2, bby1:bby2]
                unlabeled_batch_2[:, :, bbx1:bbx2, bby1:bby2] = unlabeled_batch_2[rand_index, :, bbx1:bbx2, bby1:bby2]
                outputs1_unlabeled[:, :, bbx1:bbx2, bby1:bby2] = outputs1_unlabeled[rand_index, :, bbx1:bbx2, bby1:bby2]
                outputs2_unlabeled[:, :, bbx1:bbx2, bby1:bby2] = outputs2_unlabeled[rand_index, :, bbx1:bbx2, bby1:bby2]

                unlabeled_outputs_1, _ = model1(unlabeled_batch_1)
                unlabeled_outputs_2, _, _, _ = model2(unlabeled_batch_2)                
                
                pseudo_outputs1 = torch.argmax(torch.softmax(outputs1_unlabeled, dim=1), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(torch.softmax(outputs2_unlabeled, dim=1), dim=1, keepdim=False)

                pseudo_supervision1 = ce_loss(unlabeled_outputs_1, pseudo_outputs2) 
                pseudo_supervision2 = ce_loss(unlabeled_outputs_2, pseudo_outputs1) 
                
            else:
                pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
                pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

                       
            edge_mask = get_pyramid(label_batch.unsqueeze(1)[:args.labeled_bs])
            loss_levels = []
            for y_hat_el, y in zip(edge_predictions_list[:args.labeled_bs], edge_mask):
                loss_levels.append(edge_loss(y_hat_el[:args.labeled_bs], y))          
            loss_edge_final = edge_loss(edge_combined_prediction[:args.labeled_bs], edge_mask[0])
            loss_deep_super = torch.sum(torch.stack(loss_levels))


            model1_loss = loss1_sup + consistency_weight * (pseudo_supervision1 + 0.1 * (loss_contr_labeled_1 + loss_contr_unlabeled_1))
            model2_loss = loss2_sup + consistency_weight * (pseudo_supervision2 + 0.1 * (loss_contr_labeled2 + loss_contr_unlabeled2)) + loss_edge_final + loss_deep_super

            loss = model1_loss + model2_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('model_loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('model_loss/model2_loss',
                              model2_loss, iter_num)
            
            writer.add_scalar('loss/loss_contr_labeled_1',
                              loss_contr_labeled_1, iter_num)
            writer.add_scalar('loss/loss_contr_unlabeled_1',
                              loss_contr_unlabeled_1, iter_num)
            writer.add_scalar('loss/loss_contr_labeled2',
                              loss_contr_labeled2, iter_num)
            writer.add_scalar('loss/loss_contr_unlabeled2',
                              loss_contr_unlabeled2, iter_num)
            
            writer.add_scalar('P_loss/pseudo_supervision1',
                              pseudo_supervision1, iter_num)
            writer.add_scalar('P_loss/pseudo_supervision2',
                              pseudo_supervision2, iter_num)
            

            logging.info('iteration %d : model1 loss : %f model2 loss : %f, pseudo_supervision2: %f' % 
                         (iter_num, model1_loss.item(), model2_loss.item(),pseudo_supervision2.item() ))
            if iter_num % 200 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_contr(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_edge_contr(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 10000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def get_pyramid(mask):
    with torch.no_grad():
        masks = [mask]

        for _ in range(5):
        
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask.float(), 2)       
            masks.append(small_mask)
        
        get_edge = Sobel().cuda()

        targets = []
        for mask in masks:
            sobel = get_edge(mask.float())
            sobel = torch.any(sobel > 0.5, dim=1, keepdims=True).float()                 
            targets.append(sobel)
    return targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[-1.0, 0.0, 1.0], 
                           [-2.0, 0.0, 2.0], 
                           [-1.0, 0.0, 1.0]])
        Gy = torch.tensor([[-1.0, -2.0, -1.0], 
                           [0.0, 0.0, 0.0], 
                           [1.0, 2.0, 1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.Filter = nn.Sequential(self.filter,nn.BatchNorm2d(2))

    def forward(self, img):
        x = self.Filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return torch.sigmoid(x) 
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
