# RadiomicsMapper

An open-source Python toolkit that converts 3D medical images into comprehensive radiomics feature maps using sliding window techniques. Supports parallel processing, checkpoint resumption, and batch analysis for large-scale radiomics studies.
# Key Features
🎯 Precise Feature Extraction

Voxel-wise Analysis: Per-voxel radiomics feature computation

Multi-dimensional Support: 2D and 3D feature extraction capabilities

Comprehensive Feature Set: Support for firstorder, GLCM, GLDM, GLRLM, GLSZM, NGTDM features

📊 Flexible Configuration

Parameterized Configuration: YAML-based feature extraction parameter configuration

Dynamic Binning: Customizable intensity binning parameters

Multiple Modes: Pure features, histogram-only, or combined modes

# Pipeline
Consisting of three main components:

`task.py` - Batch processing manager with resume capability and parallel execution

`run.py` - Single-image feature extraction engine with optimized voxel-wise computation

`get_range.py` - Intensity range analyzer for bin parameter optimization


Workflow：

Step 1: Calculate global min/max intensity across dataset 

`python get_range.py --data images/ --mask masks/`

Step 2: Prepare Bin Parameters

Create `bin_parameters.xlsx`

Step 3: Configure Features

Edit `params.yaml` to enable desired feature classes and settings.

Step 4: Execute Batch Processing

`python task.py --data images/ --mask masks/ --output results/ --bin_path bin_parameters.xlsx`

# Detailed Parameter Reference

```--process_nums	Number of parallel processes
--data	Input image directory
--mask	Input mask directory
--output	Output directory for results
--dim	Processing dimension (2D/3D)
--pad	Padding size for feature extraction
--bin_path	Excel file with bin parameters
--resume	Enable resume from previous execution
--get_hist	Mode: 0=features, 1=histogram, 2=both
--binWidth	Bin width for intensity discretization
--binWidthupper	Upper intensity bound
--binWidthlower	Lower intensity bound```

