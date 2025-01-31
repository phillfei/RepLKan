import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset,test_single_volume,Synapse_test_single_volume,ACDCDataset
from model.backbone import self_net
# from unet import UNet
# from unet_v2.UNet_v2 import UNetV2
from utils.utils import plot_img_and_mask
from torch.utils.data import DataLoader
from tqdm import tqdm
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                datasets = 'BASIC'):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None,full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def inference(args, model, test_save_path=None):
    # print(args.input.replace('images','labels'))
    db_test = BasicDataset(images_dir=args.input,mask_dir=args.input.replace('images','labels'), scale=args.scale, split="test_vol",augment=False,classes=args.classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_iou = []
    metric_dice = []
    metric_hd95 = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch[0].size()[2:]
        image, label, case_name = sampled_batch[0], sampled_batch[1], str(i_batch)
        # print(image.shape)
        metric_i = test_single_volume(image, label, model, classes=args.classes, patch_size=[224, 224],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        # print(metric_i)
        metric_iou.append([metric_i[i][0] for i in range(args.classes-1)])
        metric_dice.append([metric_i[i][1] for i in range(args.classes-1)])
        metric_hd95.append([metric_i[i][2] for i in range(args.classes-1)])
        logging.info('idx %d case %s mean_iou %f mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.nanmean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    # metric_list = metric_list / len(db_test)
    metric_iou = np.nanmean(np.array(metric_iou), axis=0)
    metric_dice = np.nanmean(np.array(metric_dice), axis=0)
    metric_hd95 = np.nanmean(np.array(metric_hd95), axis=0)
    print(metric_iou)
    print(metric_dice)
    print(metric_hd95)
    for i in range(args.classes):
        logging.info('Mean class %d mean_iou %f mean_dice %f mean_hd95 %f' % (i, metric_iou[i-1], metric_dice[i-1], metric_hd95[i-1]))
    performance = np.nanmean(metric_iou)
    mean_dice = np.nanmean(metric_dice)
    mean_hd95 = np.nanmean(metric_hd95)
    logging.info('Testing performance in best val model: mean_iou : %f mean_dice : %f mean_hd95 : %f' % (performance, mean_dice, mean_hd95))
    return "Testing Finished!"
def ACDCinference(args, model, test_save_path=None):
    db_test = ACDCDataset(mask_dir=args.input,images_dir = args.input, scale=args.scale, split="test_vol",augment=False,classes=args.classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch[0].size()[2:]
        image, label, case_name = sampled_batch[0], sampled_batch[1], str(i_batch)
        # print(image.shape)
        metric_i = Synapse_test_single_volume(image, label, model, classes=args.classes, patch_size=[224, 224],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_iou %f mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.classes):
        logging.info('Mean class %d mean_iou %f mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_dice = np.mean(metric_list, axis=0)[1]
    mean_hd95 = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_iou : %f mean_dice : %f mean_hd95 : %f' % (performance, mean_dice, mean_hd95))
    return "Testing Finished!"
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./output/best.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', default='./data/test/', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', default='./output/', help='Filenames of output images')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--scale', '-s', type=float, default=224,
                        help='Scale factor for the input images')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--datasets', '-d', type=str, default='ACDC', help='the name of datasets')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    # print(np.unique(out))
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)
    net = self_net(n_channels=1, n_classes=args.classes,img_dim=args.scale)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict,strict=False)
    print(sum(p.numel() for p in net.parameters()))
    logging.info('Model loaded!')
    if args.datasets == 'ISIC':
        result = inference(args, net, test_save_path=None)
    else:
        result = ACDCinference(args, net, test_save_path=None)

