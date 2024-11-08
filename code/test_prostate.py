import argparse
import os
import shutil
import logging
import sys
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
from utils import ramps
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Prostate', help='experiment_name')
parser.add_argument('--model', type=str, default='UNet_contr', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')

parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--model2', type=str, default='UNet_BMT_DCCL', help='model2_name')
args = parser.parse_args()

import cv2
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    smooth = 1e-5
    boundary_IOU = 0
    for i in range(pred.squeeze().shape[0]):
        pred_boundary = mask_to_boundary(np.uint8(pred[i].squeeze()))
        gt_boundary = mask_to_boundary(np.uint8(gt[i].squeeze()))
        boundary_inter = np.sum(pred_boundary * gt_boundary)
        boundary_union = np.sum(pred_boundary + gt_boundary) - boundary_inter
        boundary_IOU += (boundary_inter + smooth) / (boundary_union + smooth) / pred.squeeze().shape[0]


    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd , boundary_IOU

def test_single_volume(case, net1, net2, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            out_main = 0.0
            out1 = 0
            out2 = 0

            if FLAGS.model == "UNet_contr":
                out1 ,_ = net1(input)
            
            if FLAGS.model2 == "UNet_BMT_DCCL":
                out2 ,_ ,_, _ = net2(input)

            out1 = torch.softmax(out1, dim=1)
            out2 = torch.softmax(out2, dim=1)
           
            out_main = (out1 + out2)/2
            out_main = torch.argmax(out_main, dim=1).squeeze(0)
            out_main = out_main.cpu().detach().numpy()

            pred = zoom(out_main, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if "Prostate" in FLAGS.root_path:
        second_metric = first_metric
        third_metric = first_metric
    else:
        if np.sum(prediction == 2)==0:
            second_metric = 0,0,0,0,0
        else:
            second_metric = calculate_metric_percase(prediction == 2, label == 2)
        if np.sum(prediction == 3)==0:
            third_metric = 0,0,0,0,0
        else:
            third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))


    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric



def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/Prostate/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "../model/Prostate/{}_{}_labeled/predictions/".format(FLAGS.exp, FLAGS.labelnum)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=snapshot_path + "/detail.log", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    net1 = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    net2 = net_factory(net_type=FLAGS.model2, in_chns=1, class_num=FLAGS.num_classes)

    
    save_model_path1 = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))   
    save_model_path2 = os.path.join(snapshot_path, '{}_best_model2.pth'.format(FLAGS.model2))
    # save_model_path1 = os.path.join(snapshot_path, 'model1_iter_30000.pth')   
    # save_model_path2 = os.path.join(snapshot_path, 'model2_iter_30000.pth')

    net1.load_state_dict(torch.load(save_model_path1), strict=False)
    print("net1 init weight from {}".format(save_model_path1))
    net1.eval()

    net2.load_state_dict(torch.load(save_model_path2), strict=False)
    print("net2 init weight from {}".format(save_model_path2))
    net2.eval()



    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net1, net2, test_save_path, FLAGS)
        first_metric = np.asarray(first_metric)
        second_metric = np.asarray(second_metric)
        third_metric = np.asarray(third_metric)
        single_avg_metric = (first_metric + second_metric + third_metric) / 3

        logging.info("%s : " % (case))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f B-IoU %f' % ("avg_metric", single_avg_metric[0], single_avg_metric[1], single_avg_metric[2], single_avg_metric[3], single_avg_metric[4]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f B-IoU %f' % ("first_metric", first_metric[0], first_metric[1], first_metric[2], first_metric[3], first_metric[4]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f B-IoU %f' % ("second_metric", second_metric[0], second_metric[1], second_metric[2], second_metric[3], second_metric[4]))
        logging.info('%s : dice : %f  jc : %f  hd95 : %f  asd : %f B-IoU %f' % ("third_metric", third_metric[0], third_metric[1], third_metric[2], third_metric[3], third_metric[4]))
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)

    with open(test_save_path+'../avg_performance_b.txt', 'w') as f:  
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
