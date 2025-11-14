# RadiomicsMapper
An open-source Python toolkit that converts 3D medical images into comprehensive radiomics feature maps using sliding window techniques. Supports parallel processing, checkpoint resumption, and batch analysis for large-scale radiomics studies.

Project Overview
This project provides a comprehensive solution for radiomics map extraction, consisting of three main components:

`batch_run.py` - Batch processing manager with resume capability and parallel execution

`run.py` - Single-image feature extraction engine with optimized voxel-wise computation

`get_range.py` - Intensity range analyzer for parameter optimization
