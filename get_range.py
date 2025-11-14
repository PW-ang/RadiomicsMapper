import os
import SimpleITK as sitk
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', default='example/2D_mask', type=str)
    parser.add_argument('--data', default='example/2D_image', type=str)
    args = parser.parse_args()
    return args

def get_sitk_data(filename):
    sitk_image = sitk.ReadImage(filename)
    image_data = sitk.GetArrayFromImage(sitk_image)
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    return origin, spacing, direction, image_data

def save_file_with_nii(origin, spacing, direction, result, filename):
    sitk_image = sitk.GetImageFromArray(result)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    sitk.WriteImage(sitk_image, filename)

def find_min(data):
    result = 1e5
    for idx in range(data.shape[0]):
        for idy in range(data.shape[1]):
            for idz in range(data.shape[2]):
                if range(data[idx, idy, idz] != 0):
                    result = min(result, data[idx, idy, idz])
    return result


def get_data(args, filename, filename_mask):
    origin, spacing, direction, data = get_sitk_data(filename)
    _, __, ___, mask = get_sitk_data(filename_mask)
    data = data * mask
    maxnum = data.max()
    minum = find_min(data)
    return maxnum, minum
    
if __name__ == '__main__':
    args = get_parse()
    files = os.listdir(args.data)
    csv_data = pd.DataFrame({})
    all_max = 0
    all_min = 1e5
    for file in files:
        filename = file[:-7]
        data_path = os.path.join(args.data, file)
        mask_path = os.path.join(args.mask, file)
        maxnum, minum = get_data(args, data_path, mask_path)
        all_max = max(all_max, maxnum)
        all_min = min(all_min, minum)

    print(all_max, all_min)
