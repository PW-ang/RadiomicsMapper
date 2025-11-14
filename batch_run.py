import os
import warnings
warnings.filterwarnings("ignore")
import time
import argparse
import pandas as pd
import json
import glob
import yaml

def get_parse():
    """Parse command line arguments for batch processing"""
    parser = argparse.ArgumentParser(description='Batch radiomics processing with resume capability')
    parser.add_argument('--process_nums', type=int, default=2, help='Number of parallel processes')
    parser.add_argument('--mask', default='example/3D_mask', type=str, help='Directory containing mask files')
    parser.add_argument('--data', default='example/3D_image', type=str, help='Directory containing image files')
    parser.add_argument('--output', default='output', type=str, help='Output directory for results')
    parser.add_argument('--dim', type=int, default=3, help='Processing dimension (2D/3D)')
    parser.add_argument('--pad', default=2, type=int, help='Padding size for feature extraction')
    parser.add_argument('--get_hist', type=int, default=2, choices=[0, 1, 2], 
                       help='Processing mode: 0=radiomics only, 1=histogram only, 2=both')
    parser.add_argument('--bin_path', default='bin.xlsx', type=str, help='Excel file with bin parameters')
    parser.add_argument('--resume', action='store_true', help='Resume from previous execution')
    return parser.parse_args()

def log_error(error_message, log_file='error_log.txt'):
    """Log error messages with timestamp to specified file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

def load_progress_state(output_dir, log_file=None):
    """Load progress state from JSON file for resume functionality"""
    state_file = os.path.join(output_dir, '.progress_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Error loading progress state: {e}", log_file)
    return {'current_bin_idx': 0, 'processed_files': {}}

def save_progress_state(output_dir, state, log_file=None):
    """Save current progress state to JSON file"""
    state_file = os.path.join(output_dir, '.progress_state.json')
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log_error(f"Error saving progress state: {e}", log_file)

def get_yaml_data(log_file=None):
    """Load radiomics feature configuration from YAML file"""
    try:
        with open('params.yaml', 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        log_error(f"Error reading YAML config: {e}", log_file)
        return None

def check_file_completion(output_path, filename, get_hist, log_file=None):
    """Check if all required output files exist and are valid"""
    if get_hist == 1:
        # For histogram-only mode, only check histogram file
        return os.path.exists(os.path.join(output_path, 'hist.csv'))
    else:
        # For feature extraction modes, check all output files
        yaml_data = get_yaml_data(log_file)
        if yaml_data is None:
            return False
        
        # Check binWidth.csv file
        binwidth_file = os.path.join(output_path, 'binWidth.csv')
        if not os.path.exists(binwidth_file):
            log_error(f"Missing file: {binwidth_file}", log_file)
            return False
        
        try:
            df = pd.read_csv(binwidth_file)
            if df.empty:
                log_error(f"Filename {filename} not in binWidth.csv", log_file)
                return False
        except Exception as e:
            log_error(f"Error reading binWidth.csv: {e}", log_file)
            return False
        
        # Check all feature class files
        for feature_class in yaml_data['feature_class']:
            if (yaml_data['feature_class'][feature_class] is not None and 
                len(yaml_data['feature_class'][feature_class]) > 0):
                for feature_name in yaml_data['feature_class'][feature_class]:
                    feature_file = os.path.join(output_path, f"{feature_name}.nii.gz")
                    feature_cut_file = os.path.join(output_path, f"{feature_name}_cut.nii.gz")
                    
                    if not os.path.exists(feature_file):
                        log_error(f"Missing feature file: {feature_file}", log_file)
                        return False
                    if not os.path.exists(feature_cut_file):
                        log_error(f"Missing cropped feature file: {feature_cut_file}", log_file)
                        return False
        
        # For histogram+features mode, also check histogram files
        if get_hist == 2:
            hist_file = os.path.join(output_path, 'hist.csv')
            if not os.path.exists(hist_file):
                log_error(f"Missing histogram file: {hist_file}", log_file)
                return False
        
        return True

def cleanup_partial_results(output_path, get_hist, log_file=None):
    """Remove potentially incomplete results from previous runs based on get_hist mode"""
    
    # Always clean up preprocessing files regardless of mode
    preprocessing_files = [
        'pad.nii.gz', 'new_mask.nii.gz', 'cut_data.nii.gz'
    ]
    
    for file_name in preprocessing_files:
        file_path = os.path.join(output_path, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                log_error(f"Error deleting preprocessing file {file_path}: {e}", log_file)
    
    # Clean up based on get_hist mode
    if get_hist == 0:  # Only radiomics features
        # Remove feature files
        feature_patterns = ['*_cut.nii.gz', '*.nii.gz']
        for pattern in feature_patterns:
            for feature_file in glob.glob(os.path.join(output_path, pattern)):
                if any(x in feature_file for x in ['firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']):
                    try:
                        os.remove(feature_file)
                    except Exception as e:
                        log_error(f"Error deleting feature file {feature_file}: {e}", log_file)

        config_files = ['binWidth.csv']
        for file_name in config_files:
            file_path = os.path.join(output_path, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted config file: {file_path}")
                except Exception as e:
                    log_error(f"Error deleting config file {file_path}: {e}", log_file)
    
    elif get_hist == 1:  # Only histogram
        # Remove histogram-related files
        hist_files = ['hist.csv', 'hist.jpg', 'hist.nii.gz']
        for file_name in hist_files:
            file_path = os.path.join(output_path, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    log_error(f"Error deleting histogram file {file_path}: {e}", log_file)
    
    elif get_hist == 2:  # Both histogram and radiomics
        # Remove all possible output files
        all_files = [
            'hist.csv', 'hist.jpg', 'hist.nii.gz', 'binWidth.csv'
        ]
        for file_name in all_files:
            file_path = os.path.join(output_path, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    log_error(f"Error deleting file {file_path}: {e}", log_file)
        
        # Remove feature files with patterns
        feature_patterns = ['*_cut.nii.gz', '*.nii.gz']
        for pattern in feature_patterns:
            for feature_file in glob.glob(os.path.join(output_path, pattern)):
                if any(x in feature_file for x in ['firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']):
                    try:
                        os.remove(feature_file)
                    except Exception as e:
                        log_error(f"Error deleting feature file {feature_file}: {e}", log_file)

def run_subprocess_safe(cmd_args, log_file=None):
    """Safely execute subprocess command using os.system"""
    try:
        cmd_str = ' '.join(cmd_args)
        exit_code = os.system(cmd_str)
        
        if exit_code == 0:
            return True, "Execution successful"
        else:
            error_msg = f"Subprocess failed with exit code: {exit_code}"
            log_error(error_msg, log_file)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Subprocess exception: {e}"
        log_error(error_msg, log_file)
        return False, error_msg

def run(args):
    """Main batch processing function with resume capability"""
    # Initialize error logging
    error_log_file = os.path.join(args.output, 'error_log.txt')
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Error Log - Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load progress state for resume functionality
    progress_state = load_progress_state(args.output, error_log_file)
    files = [f for f in os.listdir(args.data) if not f.endswith('.npy')]
    bin_data = pd.read_excel(args.bin_path)
    
    start_bin_idx = progress_state['current_bin_idx'] if args.resume else 0
    
    # Process each bin parameter set
    for idx in range(start_bin_idx, bin_data.shape[0]):
        print(f"Processing bin parameter set {idx + 1}/{bin_data.shape[0]}")
        bin_width = (bin_data['binWidthupper'].iloc[idx] - bin_data['binWidthlower'].iloc[idx]) / bin_data['binWidth'].iloc[idx]
        output = os.path.join(args.output, f'{bin_width}')
        
        if not os.path.exists(output):
            os.makedirs(output)
        
        # Update progress state
        progress_state['current_bin_idx'] = idx
        save_progress_state(args.output, progress_state, error_log_file)
        
        bin_dicts = {}  # Accumulate histogram data across files
        
        # Process each file
        for file_idx, file in enumerate(files):
            if file.endswith('.npy'):
                continue
                
            filename = file[:-7] if '.' in file else file
            bin_key = f"bin_{idx}_file_{file_idx}"
            data_path = os.path.join(args.data, file)
            mask_path = os.path.join(args.mask, file)
            output_path = os.path.join(output, filename)
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            get_hist = args.get_hist
            
            # Check if file should be skipped
            should_skip = False
            skip_reason = ""
            
            # Skip if marked completed in state file
            if args.resume and bin_key in progress_state.get('processed_files', {}):
                if progress_state['processed_files'][bin_key] == 'completed':
                    should_skip = True
                    skip_reason = "state file marked"
            
            # Skip if output files are complete
            elif args.resume and check_file_completion(output_path, filename, get_hist, error_log_file):
                should_skip = True
                skip_reason = "output files complete"
                progress_state.setdefault('processed_files', {})[bin_key] = 'completed'
                save_progress_state(args.output, progress_state, error_log_file)
            
            # Accumulate histogram for skipped files
            if should_skip:
                print(f'Skipping completed file ({skip_reason}) [{file_idx+1}/{len(files)}]: {file}')
                try:
                    hist_file = os.path.join(output_path, 'hist.csv')
                    if os.path.exists(hist_file):
                        bin_df = pd.read_csv(hist_file)
                        for _, row in bin_df.iterrows():
                            bin_val = row['bin']
                            count_val = row['counts']
                            bin_dicts[bin_val] = bin_dicts.get(bin_val, 0) + count_val
                except Exception as e:
                    log_error(f'Error accumulating histogram for skipped file: {e}', error_log_file)
                continue
            
            # Clean up partial results for files to be processed
            if args.resume:
                print(f'Cleaning partial results [{file_idx+1}/{len(files)}]: {file}')
                cleanup_partial_results(output_path, get_hist, error_log_file)
            
            print(f'Starting processing [{file_idx+1}/{len(files)}]: {data_path}')
            print(f"Bin parameters: width={bin_data['binWidth'].iloc[idx]}, upper={bin_data['binWidthupper'].iloc[idx]}, lower={bin_data['binWidthlower'].iloc[idx]}")
            
            try:
                # Build command for single file processing
                cmd_args = [
                    'python', 'run.py',
                    '--process_nums', str(args.process_nums),
                    '--mask_filename', mask_path,
                    '--filename', data_path,
                    '--output', output_path,
                    '--dim', str(args.dim),
                    '--pad', str(args.pad),
                    '--get_hist', str(get_hist),
                    '--binWidth', str(bin_data["binWidth"].iloc[idx]),
                    '--binWidthupper', str(bin_data["binWidthupper"].iloc[idx]),
                    '--binWidthlower', str(bin_data["binWidthlower"].iloc[idx])
                ]
                
                # Execute processing
                success, message = run_subprocess_safe(cmd_args, error_log_file)
                
                if success:
                    # Accumulate histogram for newly processed file
                    try:
                        hist_file = os.path.join(output_path, 'hist.csv')
                        if os.path.exists(hist_file):
                            bin_df = pd.read_csv(hist_file)
                            for _, row in bin_df.iterrows():
                                bin_val = row['bin']
                                count_val = row['counts']
                                bin_dicts[bin_val] = bin_dicts.get(bin_val, 0) + count_val
                    except Exception as e:
                        log_error(f'Error accumulating histogram for new file: {e}', error_log_file)
                    
                    # Mark as completed
                    progress_state.setdefault('processed_files', {})[bin_key] = 'completed'
                    save_progress_state(args.output, progress_state, error_log_file)
                    print(f'Successfully completed [{file_idx+1}/{len(files)}]: {file}')
                else:
                    log_error(f'Processing failed [{file_idx+1}/{len(files)}]: {file}, reason: {message}', error_log_file)
                    
            except Exception as e:
                log_error(f'Processing exception [{file_idx+1}/{len(files)}]: {file}, error: {e}', error_log_file)
            
            # Periodically save histogram summary
            try:
                if get_hist >= 1 and bin_dicts:
                    hist_summary_file = os.path.join(output, 'hist.csv')
                    hist_df = pd.DataFrame(bin_dicts, index=[0])
                    hist_df.to_csv(hist_summary_file, index=False)
                    print(f'Updated histogram summary: {hist_summary_file}')
            except Exception as e:
                log_error(f'Error saving histogram summary: {e}', error_log_file)
        
        # Save final histogram summary for this bin set
        try:
            if get_hist >= 1 and bin_dicts:
                hist_summary_file = os.path.join(output, 'hist.csv')
                hist_df = pd.DataFrame(bin_dicts, index=[0])
                hist_df.to_csv(hist_summary_file, index=False)
                print(f'Final histogram summary saved: {hist_summary_file}')
        except Exception as e:
            log_error(f'Error saving final histogram summary: {e}', error_log_file)
        
        print(f"Completed bin parameter set {idx + 1}")
    
    # Clean up progress state file after completion
    state_file = os.path.join(args.output, '.progress_state.json')
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
        except Exception as e:
            log_error(f'Error deleting state file: {e}', error_log_file)
    
    # Log completion
    with open(error_log_file, 'a', encoding='utf-8') as f:
        f.write(f"\nAll processing completed - End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("All processing completed!")

if __name__ == '__main__':
    args = get_parse()
    
    # Handle resume mode
    if args.resume:
        print("Resume mode: Continuing from previous execution")
        state_file = os.path.join(args.output, '.progress_state.json')
        if not os.path.exists(state_file):
            print("No progress state file found, starting from beginning")
    else:
        print("Normal mode: Starting fresh execution")
        state_file = os.path.join(args.output, '.progress_state.json')
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
            except Exception as e:
                log_error(f"Error deleting state file: {e}")
    
    run(args)