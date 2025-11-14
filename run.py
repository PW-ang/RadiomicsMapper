import os
import numpy as np
import SimpleITK as sitk
from radiomics_customize import firstorder, glcm, gldm, glrlm, glszm, ngtdm
import matplotlib.pyplot as plt
import six
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from time import time
import argparse
import yaml
import pandas as pd
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import copy

def get_parse():
    """Parse command line arguments for radiomics feature extraction"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_nums', type=int, default=2, help='Number of parallel processes')
    parser.add_argument('--mask_filename', default='example/3D_mask/001.nii.gz', type=str, help='Input mask file path')
    parser.add_argument('--filename', default='example/3D_image/001.nii.gz', type=str, help='Input image file path')
    parser.add_argument('--output', default='output', type=str, help='Output directory path')
    parser.add_argument('--dim', type=int, default=3, help='Dimension (2D/3D) for processing')
    parser.add_argument('--pad', default=2, type=int, help='Padding size defining neighborhood (1=3x3x3, 2=5x5x5, etc.)')
    parser.add_argument('--get_hist', default=0, type=int, help='Histogram generation mode (0: cal radiomics, 1: get hist, 2: get hist and cal radiomics)')
    parser.add_argument('--binWidth', default=100, type=float, help='Bin width for intensity discretization')
    parser.add_argument('--binWidthupper', default=4100, type=float, help='Upper bound for intensity range')
    parser.add_argument('--binWidthlower', default=300, type=float, help='Lower bound for intensity range')
    return parser.parse_args()

def get_sitk_data(filename):
    """Load image data and metadata using SimpleITK"""
    sitk_image = sitk.ReadImage(filename)
    image_data = sitk.GetArrayFromImage(sitk_image)
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    return origin, spacing, direction, image_data

def save_file_with_nii(origin, spacing, direction, result, filename):
    """Save numpy array as NIfTI file with original metadata"""
    sitk_image = sitk.GetImageFromArray(result)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    sitk.WriteImage(sitk_image, filename)

def get_new_mask(args, mask):
    """Generate new mask by propagating existing mask values to neighboring voxels"""
    data = np.copy(mask)
    epochs = 3
    while epochs > 0:
        pad = epochs
        for idx in range(data.shape[0]):
            for idy in range(data.shape[1]):
                for idz in range(data.shape[2]):
                    if mask[idx][idy][idz] == 0:
                        # Propagate mask values from neighboring voxels in 6-connectivity
                        if idz + pad < data.shape[2] and mask[idx][idy][idz + pad] == 1:
                            data[idx, idy, idz] = data[idx, idy, idz + pad]
                        elif idz - pad >= 0 and mask[idx][idy][idz - pad] == 1:
                            data[idx, idy, idz] = data[idx, idy, idz - pad]
                        elif idy + pad < data.shape[1] and mask[idx][idy + pad][idz] == 1:
                            data[idx, idy, idz] = data[idx, idy + pad, idz]
                        elif idy - pad >= 0 and mask[idx][idy - pad][idz] == 1:
                            data[idx, idy, idz] = data[idx, idy - pad, idz]
                        if args.dim == 3:  # For 3D, also check through-plane neighbors
                            if idx + pad < data.shape[0] and mask[idx + pad][idy][idz] == 1:
                                data[idx, idy, idz] = data[idx + pad, idy, idz]
                            elif idx - pad >= 0 and mask[idx - pad][idy][idz] == 1:
                                data[idx, idy, idz] = data[idx - pad, idy, idz]
        epochs = epochs - 1
    return data

def pad_data(args, data, mask):
    """Pad image data by propagating intensity values from mask regions"""
    epochs = 3
    while epochs > 0:
        pad = epochs
        for idx in range(data.shape[0]):
            for idy in range(data.shape[1]):
                for idz in range(data.shape[2]):
                    if mask[idx][idy][idz] == 0:
                        # Fill non-mask voxels with values from neighboring mask voxels
                        if idz + pad < data.shape[2] and mask[idx][idy][idz + pad] == 1:
                            data[idx, idy, idz] = data[idx, idy, idz + pad]
                        elif idz - pad >= 0 and mask[idx][idy][idz - pad] == 1:
                            data[idx, idy, idz] = data[idx, idy, idz - pad]
                        elif idy + pad < data.shape[1] and mask[idx][idy + pad][idz] == 1:
                            data[idx, idy, idz] = data[idx, idy + pad, idz]
                        elif idy - pad >= 0 and mask[idx][idy - pad][idz] == 1:
                            data[idx, idy, idz] = data[idx, idy - pad, idz]
                        elif idx + pad < data.shape[0] and mask[idx + pad][idy][idz] == 1:
                            data[idx, idy, idz] = data[idx + pad, idy, idz]
                        elif idx - pad >= 0 and mask[idx - pad][idy][idz] == 1:
                            data[idx, idy, idz] = data[idx - pad, idy, idz]
        epochs = epochs - 1
    return data

def cut_pic(data, mask):
    """Crop image to bounding box containing mask region"""
    Up = 10000
    Left = 10000
    Down = -1
    Right = -1
    result = []
    new_mask = []
    map_z = {}  # Mapping from cropped slice index to original slice index
    icount_idz = 0
    
    # Find bounding box coordinates
    for idz in range(data.shape[0]):
        if mask[idz].sum() == 0:  # Skip empty slices
            continue
        map_z[icount_idz] = idz
        icount_idz = icount_idz + 1
        for idx in range(data.shape[1]):
            for idy in range(data.shape[2]):
                if mask[idz][idx][idy] == 1:
                    Up = min(Up, idx)
                    Left = min(Left, idy)
                    Down = max(Down, idx)
                    Right = max(Right, idy)
    
    # Extract cropped region
    for idz in range(data.shape[0]):
        if mask[idz].sum() == 0:
            continue
        result.append(data[idz, Up:Down+1, Left:Right+1])
        new_mask.append(mask[idz, Up:Down+1, Left:Right+1])
        
    return map_z, np.array(result), Up, Left, np.array(new_mask)

def get_data(args, filename, filename_mask):
    """Load and preprocess image and mask data"""
    origin, spacing, direction, data = get_sitk_data(filename)
    _, __, ___, mask = get_sitk_data(filename_mask)
    
    # Calculate initial bin width based on non-zero intensity mean
    origin_binwidth = data[data != 0].mean() / 10
    
    # Preprocess data
    data = pad_data(args, data, mask)
    os.makedirs(args.output, exist_ok=True)
    save_file_with_nii(origin, spacing, direction, data, os.path.join(args.output, 'pad.nii.gz'))
    
    new_mask = get_new_mask(args, mask)
    save_file_with_nii(origin, spacing, direction, new_mask, os.path.join(args.output, 'new_mask.nii.gz'))
    
    map_z, cut_data, Up, Left, new_mask = cut_pic(data, new_mask)
    save_file_with_nii(origin, spacing, direction, cut_data, os.path.join(args.output, 'cut_data.nii.gz'))
    
    return origin, spacing, direction, data, mask, origin_binwidth, map_z, cut_data, Up, Left, new_mask

def get_yaml_data():
    """Load radiomics feature configuration from YAML file"""
    with open('params.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data

def show(result, feature_type):
    """Format feature results with prefix for output"""
    tmp = {}
    for key, val in six.iteritems(result):
        tmp['original_{}_{}'.format(feature_type, key)] = val
    return tmp

def extract_feature_unit(args, yaml_data, c, p, q, sub_img, origin_binwidth):
    """Extract radiomics features for a single voxel neighborhood"""
    import copy
    padding = args.pad
    features_temp = {}
    C, H, W = sub_img.shape
    mask = np.copy(sub_img)
    
    # Create local mask around target voxel
    if args.dim == 3:
        mask[:, :, :] = 0
        mask[max(0, c - padding):min(C, c + padding + 1),
             max(0, p - padding):min(H, p + padding + 1), 
             max(0, q - padding):min(W, q + padding + 1)] = 1
    else:
        mask[:, :, :] = 0
        mask[: ,max(0, p - padding):min(H, p + padding + 1), 
                max(0, q - padding):min(W, q + padding + 1)] = 1
                
    # Convert to SimpleITK images for radiomics extraction
    img_ex = sitk.GetImageFromArray(sub_img)
    mask_ex = sitk.GetImageFromArray(mask)
    settings = copy.deepcopy(yaml_data['settings'])
    binWidth_dict = {}
    
    # Set default bin width if not specified
    if settings.get('binWidth') is None and settings.get('binCount') is None and settings.get('binWidthupper') is None and settings.get('binWidthlower') is None:
        settings['binWidth'] = origin_binwidth

    # Extract firstorder features if enabled
    if yaml_data['feature_class'].get('firstorder') is not None:
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['firstorder']]
        for feature in features:
            if feature == 'StandardDeviation':  # Skip StandardDeviation feature
                continue
            firstOrderFeatures.enableFeatureByName(feature, True)
        result_firstorder = firstOrderFeatures.execute()
        features_temp["first"] = show(result_firstorder, 'firstorder')
        binWidth_dict['firstorder'] = settings['binWidth']
    
    # Extract GLCM features if enabled
    if yaml_data['feature_class'].get('glcm') is not None:
        glcmFeatures = glcm.RadiomicsGLCM(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['glcm']]
        for idx, feature in enumerate(features):
            if idx == 21:  # Skip specific GLCM feature
                continue
            glcmFeatures.enableFeatureByName(feature, True)
        result_glcm = glcmFeatures.execute()
        features_temp["glcm"] = show(result_glcm, 'glcm')
        binWidth_dict['glcm'] = settings['binWidth']
    
    # Extract GLDM features if enabled
    if yaml_data['feature_class'].get('gldm') is not None:
        gldmFeatures = gldm.RadiomicsGLDM(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['gldm']]
        for feature in features:
            gldmFeatures.enableFeatureByName(feature, True)
        result_gldm = gldmFeatures.execute()
        features_temp["gldm"] = show(result_gldm, 'gldm')
        binWidth_dict['gldm'] = settings['binWidth']
    
    # Extract GLRLM features if enabled
    if yaml_data['feature_class'].get('glrlm') is not None:
        glrlmFeatures = glrlm.RadiomicsGLRLM(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['glrlm']]
        for feature in features:
            glrlmFeatures.enableFeatureByName(feature, True)
        result_glrlm = glrlmFeatures.execute()
        features_temp["glrlm"] = show(result_glrlm, 'glrlm')
        binWidth_dict['glrlm'] = settings['binWidth']
    
    # Extract GLSZM features if enabled
    if yaml_data['feature_class'].get('glszm') is not None:
        glszmFeatures = glszm.RadiomicsGLSZM(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['glszm']]
        for feature in features:
            glszmFeatures.enableFeatureByName(feature, True)
        result_glszm = glszmFeatures.execute()
        features_temp["glszm"] = show(result_glszm, 'glszm')
        binWidth_dict['glszm'] = settings['binWidth']
    
    # Extract NGTDM features if enabled
    if yaml_data['feature_class'].get('ngtdm') is not None:
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(img_ex, mask_ex, **settings)
        features = [fea.split('_')[-1] for fea in yaml_data['feature_class']['ngtdm']]
        for feature in features:
            ngtdmFeatures.enableFeatureByName(feature, True)
        result_ngtdm = ngtdmFeatures.execute()
        features_temp["ngtdm"] = show(result_ngtdm, 'ngtdm')
        binWidth_dict['ngtdm'] = settings['binWidth']
    
    # Return results with position information
    results = {
        'result': features_temp,
        'c': c,
        'p': p,
        'q': q,
        'binWidth': binWidth_dict
    }
    return results

def worker_process(chunk_positions, shm_name, shape, dtype_str, args_serializable, yaml_data, origin_binwidth):
    """
    Worker process for parallel feature extraction using shared memory
    
    Args:
        chunk_positions: List of (c,p,q) coordinate tuples to process
        shm_name: Shared memory block name
        shape: Shape of the image data array
        dtype_str: Data type string of the image data
        args_serializable: Picklable arguments namespace
        yaml_data: Feature configuration dictionary
        origin_binwidth: Default bin width for intensity discretization
    """
    # Attach to existing shared memory block
    shm = shared_memory.SharedMemory(name=shm_name)
    dtype = np.dtype(dtype_str)
    data_view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    results_chunk = []
    
    # Process each position in the chunk
    for (c, p, q) in chunk_positions:
        try:
            res = extract_feature_unit(args_serializable, yaml_data, c, p, q, data_view, origin_binwidth)
            results_chunk.append(res)
        except Exception as e:
            # Return error information without crashing the process
            results_chunk.append({'error': str(e), 'c': c, 'p': p, 'q': q})
    
    # Close shared memory connection (don't unlink - main process handles cleanup)
    shm.close()
    return results_chunk

def chunkify(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def process_run_improved(args, data, yaml_data, tasks, origin_binwidth):
    """
    Parallel radiomics feature extraction using shared memory and process pool
    
    Args:
        args: Command line arguments
        data: 3D image data array (C,H,W)
        yaml_data: Feature configuration dictionary
        tasks: List of (c,p,q) coordinates to process
        origin_binwidth: Default bin width for intensity discretization
    """
    start_time = time()
    manager_results = {}
    
    # Initialize result arrays for all enabled features
    for keys in yaml_data['feature_class'].keys():
        if yaml_data['feature_class'][keys] != None:
            for key in yaml_data['feature_class'][keys]:
                manager_results[key] = np.zeros(data.shape, dtype=float)

    # Create shared memory for image data
    dtype = data.dtype
    shape = data.shape
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    
    try:
        # Copy data to shared memory
        shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shm_array[:] = data[:]

        total_tasks = len(tasks)
        if total_tasks == 0:
            print("No tasks to run.")
            return manager_results

        # Calculate optimal chunk size for parallel processing
        n_workers = max(1, args.process_nums)
        target_chunks = n_workers * 6  # Aim for ~6 chunks per worker
        chunk_size = max(1, math.ceil(total_tasks / target_chunks))
        chunks = list(chunkify(tasks, chunk_size))

        # Execute tasks in parallel
        futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks to process pool
            for chunk in chunks:
                f = executor.submit(worker_process, chunk, shm.name, shape, dtype.str, args, yaml_data, origin_binwidth)
                futures.append(f)

            # Process results as they complete
            pbar = tqdm(total=total_tasks, desc="Processing voxels", unit="vox")
            for fut in as_completed(futures):
                res_chunk = fut.result()
                for item in res_chunk:
                    if 'error' in item:
                        # Skip errored voxels but continue processing
                        pbar.update(1)
                        continue
                    
                    # Store feature results in output arrays
                    for keys in item['result'].keys():
                        for key in item['result'][keys].keys():
                            manager_results[key][item['c'], item['p'], item['q']] = item['result'][keys][key]
                    
                    # Save binWidth information (first successful voxel only)
                    if not os.path.exists(os.path.join(args.output, 'binWidth.csv')):
                        pd.DataFrame(item['binWidth'], index=[0]).to_csv(os.path.join(args.output, 'binWidth.csv'), index=False)
                    
                    pbar.update(1)
            pbar.close()

    finally:
        # Cleanup shared memory
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    end_time = time()
    print('The file {} processes with {}s'.format(args.filename, end_time - start_time))
    return manager_results

def get_hist(yaml_data, data, origin_binwidth, filename):
    """Generate histogram data for radiomics analysis"""
    mask = np.ones(data.shape)
    img_ex = sitk.GetImageFromArray(data)
    mask_ex = sitk.GetImageFromArray(mask)
    settings = copy.deepcopy(yaml_data['settings'])
    
    if settings.get('binWidth') is None and settings.get('binCount') is None and settings.get('binWidthupper') is None and settings.get('binWidthlower') is None:
        settings['binWidth'] = origin_binwidth
        
    firstorder.RadiomicsFirstOrder(img_ex, mask_ex, filename=filename, **settings)
    
def run():
    """Main function for single image radiomics feature extraction"""
    args = get_parse()
    
    # Load and preprocess data
    origin, spacing, direction, data, mask, origin_binwidth, map_z, cut_data, Up, Left, new_mask = \
        get_data(args, args.filename, args.mask_filename)
    
    yaml_data = get_yaml_data()
    
    # Update binning parameters from command line arguments
    if args.binWidth is not None:
        yaml_data['settings']['binWidth'] = args.binWidth
    if args.binWidthupper is not None:
        yaml_data['settings']['binWidthupper'] = args.binWidthupper
    if args.binWidthlower is not None:
        yaml_data['settings']['binWidthlower'] = args.binWidthlower

    # Generate task list for voxels within mask region
    tasks = []
    for idx in range(cut_data.shape[0]):
        for idy in range(cut_data.shape[1]):
            for idz in range(cut_data.shape[2]):
                if mask[map_z[idx], idy + Up, idz + Left] == 1:
                    tasks.append((idx, idy, idz))
    print("Total voxels:", len(tasks))

    # Handle histogram generation mode
    if args.get_hist > 0:
        get_hist(yaml_data, cut_data, origin_binwidth, args.filename)
        hist_data = np.load(args.filename + '.npy')
        hist_data = hist_data * new_mask  # Apply mask to histogram data
        save_file_with_nii(origin, spacing, direction, hist_data, os.path.join(args.output, 'hist.nii.gz'))
        
        # Analyze and save histogram statistics
        unique_elements, counts = np.unique(hist_data, return_counts=True)
        data_csv = pd.DataFrame({'bin': unique_elements, 'counts': counts})
        data_csv.to_csv(os.path.join(args.output, 'hist.csv'), index=False)
        
        # Generate histogram visualization
        hist_data = hist_data[hist_data != 0]
        plt.figure()
        if yaml_data['settings']['binWidth'] is not None:
            plt.hist(hist_data.reshape(-1) * yaml_data['settings']['binWidth'], 
                    bins = [_ * yaml_data['settings']['binWidth'] for _ in range(int(unique_elements.max()))])
            plt.xlim(yaml_data['settings']['binWidthlower'], int(unique_elements.max()) * yaml_data['settings']['binWidth'])
        else:
            plt.hist(hist_data.reshape(-1), bins=len(unique_elements))
        plt.savefig(os.path.join(args.output, 'hist.jpg'), dpi=300)
        
        # Cleanup temporary files if only histogram was requested
        if args.get_hist == 1:
            if os.path.exists(args.filename + '.npy'):
                os.remove(args.filename + '.npy')
            return

    # Main feature extraction
    results = process_run_improved(args, cut_data, yaml_data, tasks, origin_binwidth)

    # Save feature maps
    for keys in yaml_data['feature_class'].keys():
        if yaml_data['feature_class'][keys] != None:
            for key in yaml_data['feature_class'][keys]:
                # Save cropped feature map
                save_file_with_nii(origin, spacing, direction, results[key], 
                                 os.path.join(args.output, key + '_cut.nii.gz'))
                
                # Save full-size feature map
                tmp = np.zeros(data.shape)
                for idx, idy, idz in tasks:
                    tmp[map_z[idx], idy + Up, idz + Left] = results[key][idx, idy, idz]
                save_file_with_nii(origin, spacing, direction, tmp, 
                                 os.path.join(args.output, key + '.nii.gz'))
    
    # Cleanup temporary files
    if os.path.exists(args.filename + '.npy'):
        os.remove(args.filename + '.npy')

if __name__ == '__main__':
    run()