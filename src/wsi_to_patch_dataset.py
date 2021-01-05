import cv2
import glob
import argparse
import os
import skimage.io
import skimage.transform
from shutil import copy2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import multiprocessing

def contains_tissue(image):
    colour_threshold = 200
    percentage_white_threshold = 0.8
    blurr_threshold = 70

    white = (255, 255, 255)
    grey = (colour_threshold, colour_threshold, colour_threshold)
    resolution = (512, 512)

    mask = cv2.inRange(image, grey, white)
    white_pixels = np.sum(mask==255)
    not_white = (white_pixels / (resolution[0] * resolution[1]) < percentage_white_threshold)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    not_blurry = fm > blurr_threshold

    # copy file if percentage of white is below threshold
    if not_white and not_blurry:
        return True
    else: # else only copy for debug
        # if not not_white:
        #     print('Patch white!')
        # elif not not_blurry:
        #     print('Patch blurry!')
        return False

def init_patch_df(existing_patch_df='None'):
    if existing_patch_df == 'None':
        df = pd.DataFrame(columns=['image_name', 'NC', 'G3', 'G4', 'G5', 'unlabeled'])
    else:
        df = pd.read_excel(existing_patch_df)
    return df

def check_max_class(patch_mask, data_provider):
    threshold_non_cancerous = 0.95
    label = None
    assert np.all(patch_mask[:,:,1] == 0) # check if other channels always zero
    assert np.all(patch_mask[:,:,2] == 0) # check if other channels always zero
    patch_mask = patch_mask[:,:,0]
    num_pixels = patch_mask.size
    num_background = np.count_nonzero(patch_mask == 0)

    if num_background > num_pixels/2:
        label = 'background'
    elif data_provider=='karolinska':
        num_nc = np.count_nonzero(patch_mask == 1)
        num_c = np.count_nonzero(patch_mask == 2)
        num_max = max([num_nc, num_c])
        if num_nc + num_background> threshold_non_cancerous*num_pixels:
            label = 'NC'
        else:
            label = 'unlabeled'
    elif data_provider=='radboud':
        num_nc = np.count_nonzero(np.logical_or(patch_mask==1, patch_mask==2))
        num_gg_3 = np.count_nonzero(patch_mask==3)
        num_gg_4 = np.count_nonzero(patch_mask==4)
        num_gg_5 = np.count_nonzero(patch_mask==5)
        num_max = max([num_gg_3, num_gg_4, num_gg_5])
        if num_nc + num_background> threshold_non_cancerous*num_pixels:
            label = 'NC'
        elif num_gg_3 == num_max:
            label = 'G3'
        elif num_gg_4 == num_max:
            label = 'G4'
        elif num_gg_5 == num_max:
            label = 'G5'
    assert label is not None

    return label

def create_patch_df_row(patch_mask, wsi_df_row, patch_name):
    is_background = False
    patch_df = None

    label = check_max_class(patch_mask, wsi_df_row['data_provider'].array[0])
    if label == 'background':
        is_background = True
    else:
        patch_df = pd.DataFrame([[patch_name, 0, 0, 0, 0, 0]],
                                columns=['image_name', 'NC', 'G3', 'G4', 'G5', 'unlabeled'])
        patch_df[label] = 1
        assert np.count_nonzero(np.array(patch_df[['NC', 'G3', 'G4', 'G5', 'unlabeled']])) == 1

    return patch_df, is_background

def read_wsi_and_mask(args, wsi_name, wsi_df):
    wsi = None
    mask = None
    wsi_path = os.path.join(args.data_dir, 'train_images', wsi_name + '.tiff')

    # wsi = cv2.imread(wsi_path)
    wsi_df_row = wsi_df[wsi_df['image_id'] == wsi_name]
    wsi_tiff = skimage.io.MultiImage(wsi_path)
    if len(wsi_tiff) == 1:
        wsi = wsi_tiff[0]
        assert len(wsi_df_row) == 1
        # if wsi_df_row['Gleason_primary'].array[0] != '0':
        mask_path = os.path.join(args.data_dir, 'train_label_masks', wsi_name + '_mask.tiff')
        mask = skimage.io.MultiImage(mask_path)[0]
    elif len(wsi_tiff) == 0:
        raise Warning('WSI tiff is empty: ' + wsi_name)
    else:
        raise Warning('WSI ' + wsi_name + ' has an unexpected number of slice: ' + str(len(wsi_tiff)))

    return wsi, mask, wsi_df_row

def calc_num_patches(wsi, resolution):
    h, w, _ = wsi.shape
    num_patches_per_row = int(2*np.floor((h/resolution)) - 1)
    num_patches_per_column = int(2*np.floor((w/resolution)) - 1)
    return num_patches_per_row, num_patches_per_column


def slice_image(args, wsi_name, wsi_df, output_dir, dataframes_only, index, return_dict):
    print('Slice WSI ' + wsi_name)
    wsi, mask, wsi_df_row = read_wsi_and_mask(args, wsi_name, wsi_df)
    width = int(wsi.shape[1] * args.resize_factor)
    height = int(wsi.shape[0] * args.resize_factor)
    dim = (width, height)
    wsi = cv2.resize(wsi, dim, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
    resolution = args.patch_resolution
    num_patches_per_row, num_patches_per_column = calc_num_patches(wsi, resolution)

    complete_patch_df = init_patch_df()
    if wsi is not None:
        for row in range(num_patches_per_row):
            for column in range(num_patches_per_column):

                start_y = int(row*(resolution/2))
                start_x = int(column*(resolution/2))
                patch = wsi[start_y:start_y+resolution, start_x:start_x+resolution]
                patch_mask = mask[start_y:start_y+resolution, start_x:start_x+resolution]

                name = wsi_name + '_' + str(row) + '_' + str(column) + '.jpg'
                patch_df, is_background = create_patch_df_row(patch_mask, wsi_df_row, name)
                if is_background is False:
                    complete_patch_df = pd.concat([complete_patch_df, patch_df], ignore_index=True)
                    if not dataframes_only:
                        # patch = skimage.transform.rescale(patch, scale=0.5, multichannel=True, anti_aliasing=True)
                        # patch_mask = skimage.transform.rescale(patch_mask, scale=0.5, multichannel=True, anti_aliasing=True)
                        skimage.io.imsave(os.path.join(output_dir,'images', name), patch)
                        skimage.io.imsave(os.path.join(output_dir, 'masks', name), patch_mask*50, check_contrast=False)

    return_dict[index] = complete_patch_df


def main(args):
    modes = ['train', 'val', 'test']
    wsi_df = pd.read_csv(args.wsi_dataframe)
    wsi_df['Gleason_primary'] = wsi_df['gleason_score'].str.split('+').str[0]
    wsi_df['Gleason_secondary'] = wsi_df['gleason_score'].str.split('+').str[1]

    for mode in modes:
        print('Split: ' + mode)
        mode_wsi_df = wsi_df[wsi_df['Partition']==mode]
        wsi_list = mode_wsi_df['image_id'].array

        if args.number_wsi != 'all' and mode =='train':
            random.seed(42)
            wsi_list = random.sample(wsi_list, int(args.number_wsi))

        number_of_wsi = len(wsi_list)
        output_patch_dir = os.path.join(args.output_dir, 'images')
        output_mask_dir = os.path.join(args.output_dir, 'masks')
        print('number_of_wsi  ', number_of_wsi)

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(output_patch_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        df = init_patch_df(args.existing_patch_df)

        filtered_wsi = []
        mp = args.multiprocesses

        for i in range(0, len(wsi_list), mp):
            print(' Spawn new processes')
            print('Working on index ' +str(i) + ' of ' + str(len(wsi_list)))
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            processes = []
            for j in range(mp):
                index = i+j
                if index == len(wsi_list):
                    break
                else:
                    wsi_name = wsi_list[index]
                    p = multiprocessing.Process(target=slice_image, args=(args, wsi_name, wsi_df, args.output_dir, args.dataframes_only, index, return_dict))
                    processes.append(p)
                    p.start()
                    # patch_df = slice_image(args, wsi_name, wsi_df, args.output_dir, args.dataframes_only, index, return_dict)
            for p in processes:
                p.join()
            for index in return_dict:
                patch_df = return_dict[index]
                if len(patch_df)==0:
                    print('All patches of the WSI have been filtered out. WSI:' + str(wsi_name))
                    filtered_wsi.append(wsi_name)

                df = pd.concat([df, patch_df])
        df.to_csv(os.path.join(args.output_dir, mode+'_patches.csv'), index=False)

        if len(filtered_wsi) > 0:
            print('The following WSI have been filtered out completely because of whiteness or blur:')
            print(filtered_wsi)
    copy2("artifacts/README.txt", os.path.join(args.output_dir))
    copy2("artifacts/train.csv", os.path.join(args.output_dir, 'wsi_labels.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-dd", type=str, default="/data/BasesDeDatos/Panda")
    # parser.add_argument("--data_dir", "-dd", type=str, default="/home/arne/datasets/Panda")
    parser.add_argument("--wsi_dataframe", "-wd", type=str, default="./artifacts/train_dummy.csv")
    parser.add_argument("--wsi_list", "-wl", type=str, default="all")
    parser.add_argument("--existing_patch_df", "-ep", type=str, default="None")

    parser.add_argument("--output_dir", "-o", type=str, default="/data/BasesDeDatos/Panda/Panda_patches_resized")
    # parser.add_argument("--output_dir", "-o", type=str, default="/home/arne/datasets/Panda/Panda_patches_resized")
    parser.add_argument("--number_wsi", "-n", type=str, default="all")
    parser.add_argument("--dataframes_only", "-do", action='store_true')
    parser.add_argument("--multiprocesses", "-mp", type=int, default=10)

    parser.add_argument("--patch_overlap", "-po", action='store_true')
    parser.add_argument("--patch_resolution", "-pr", type=int, default=512)
    parser.add_argument("--resize_factor", "-rf", type=float, default=0.5)
    parser.add_argument("--debug", "-d", action='store_true')
    args = parser.parse_args()
    print('Arguments:')
    print(args)
    main(args)