# RadiomicsMapper
An open-source Python toolkit that converts 3D medical images into comprehensive radiomics feature maps using sliding window techniques. Supports parallel processing, checkpoint resumption, and batch analysis for large-scale radiomics studies.

Project Overview
This project provides a comprehensive solution for radiomics map extraction, consisting of three main components:
`task.py` - Batch processing manager with resume capability and parallel execution
`run.py` - Single-image feature extraction engine with optimized voxel-wise computation
`get_range.py` - Intensity range analyzer for bin parameter optimization

🎯 Precise Feature Extraction
Voxel-wise Analysis: Per-voxel radiomics feature computation
Multi-dimensional Support: 2D and 3D feature extraction capabilities
Comprehensive Feature Set: Support for firstorder, GLCM, GLDM, GLRLM, GLSZM, NGTDM features

📊 Flexible Configuration
Parameterized Configuration: YAML-based feature extraction parameter configuration
Dynamic Binning: Customizable intensity binning parameters
Multiple Modes: Pure features, histogram-only, or combined modes

Quick Start
1. Intensity Range Analysis
`bash
python get_range.py --data images/ --mask masks/`
Output: Global intensity range for bin parameter optimization

2. Single Image Processing
`bash
python run.py \
    --filename image.nii.gz \
    --mask_filename mask.nii.gz \
    --output results/ \
    --process_nums 8 \
    --dim 3 \
    --pad 2 \
    --get_hist 2`
3. Batch Processing
`bash
python batch_run.py \
    --data images/ \
    --mask masks/ \
    --output batch_results/ \
    --process_nums 4 \
    --bin_path bin_parameters.xlsx \
    --resume`

Complete Workflow
Step 1: Analyze Intensity Range
`bash
# Calculate global min/max intensity across dataset
python get_range.py --data images/ --mask masks/`
# Output example: 400.0 -100.0 (use these for binWidthupper/binWidthlower)
Step 2: Prepare Bin Parameters
Create bin_parameters.xlsx:

binWidth	binWidthupper	binWidthlower
25	400	-100
50	400	-100
100	400	-100
Step 3: Configure Features
Edit params.yaml to enable desired feature classes and settings.

Step 4: Execute Batch Processing
`bash
python batch_run.py --data images/ --mask masks/ --output results/ --bin_path bin_parameters.xlsx`
Detailed Parameter Reference
`run.py` Parameters
`Parameter	Type	Default	Description
--process_nums	int	10	Number of parallel processes
--mask_filename	str	'mask'	Input mask file path
--filename	str	'image'	Input image file path
--dim	int	3	Processing dimension (2/3)
--output	str	'output'	Output directory
--pad	int	2	Neighborhood size (1=3×3×3, 2=5×5×5)
--get_hist	int	0	Mode: 0=features, 1=histogram, 2=both
--binWidth	float	None	Bin width for intensity discretization
--binWidthupper	float	None	Upper intensity bound
--binWidthlower	float	None	Lower intensity bound`
batch_run.py Parameters
`Parameter	Type	Default	Description
--process_nums	int	2	Number of parallel processes
--data	str	'image'	Input image directory
--mask	str	'mask'	Input mask directory
--output	str	'output'	Output directory for results
--dim	int	3	Processing dimension (2D/3D)
--pad	int	1	Padding size for feature extraction
--bin_path	str	'bin.xlsx'	Excel file with bin parameters
--resume	flag	False	Enable resume from previous execution`
Processing Modes
`--get_hist` 0: Extract radiomics features only

`--get_hist` 1: Generate histogram statistics only

`--get_hist` 2: Extract features and generate histogram
