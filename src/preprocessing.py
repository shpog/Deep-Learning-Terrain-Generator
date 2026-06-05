"""
Preprocessing Data Pipeline for Terrain Generation.

This module handles the extraction, alignment, aggregation, and normalization
of high-resolution GeoTIFF files from WorldClim. It is optimized to process
massive global datasets (up to 30s resolution) iteratively to prevent memory
overflow, outputting a memory-mapped NumPy array for training.
"""

import os
import glob
import rasterio
import numpy as np
import torch

from config import RESOLUTION, RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_geotiff(file_path):
    """
    Reads a single GeoTIFF file and extracts the 2D array and NoData value.

    Args:
        file_path (str or pathlib.Path): The absolute or relative path to the .tif file.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The 2D array of the raster data.
            - float: The specific NoData value used in this raster.
    """
    with rasterio.open(file_path) as src:
        return src.read(1), src.nodata

def aggregate_monthly_data(file_pattern, operation='sum'):
    """
    Aggregates 12 monthly climate files into a single annual array.

    This function operates iteratively. It loads one file, applies it to a 
    running total, and deletes it from memory before loading the next. 
    This prevents RAM overflow on 30s global datasets.

    Args:
        file_pattern (str): The glob pattern to match the 12 monthly files.
        operation (str, optional): The aggregation method to use. 
            Accepts 'sum' (for precipitation) or 'mean' (for temperature). 
            Defaults to 'sum'.

    Raises:
        FileNotFoundError: If no files match the provided file_pattern.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The aggregated 2D array.
            - float: The NoData value used in the dataset.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")
    
    print(f"Aggregating {len(files)} files from {file_pattern} using '{operation}'...")
    
    first_arr, nodata_val = load_geotiff(files[0])
    valid_mask = (first_arr != nodata_val)
    
    accumulator = np.zeros(first_arr.shape, dtype=np.float32)
    accumulator[valid_mask] = first_arr[valid_mask]
    
    del first_arr 
    
    for f in files[1:]:
        arr, _ = load_geotiff(f)
        accumulator[valid_mask] += arr[valid_mask]
        del arr  # Aggressively free memory
        
    if operation == 'mean':
        accumulator[valid_mask] /= len(files)
        
    aggregated = np.full(accumulator.shape, nodata_val, dtype=np.float32)
    aggregated[valid_mask] = accumulator[valid_mask]
        
    return aggregated, nodata_val

def clean_and_normalize(array, nodata_val, fill_value=0.0):
    """
    Isolates valid data, normalizes it to [0, 1], and fills NoData gaps.

    It is crucial to calculate the min/max ONLY on valid data, otherwise
    the NoData values (e.g., -9999) will distort the scaling.

    Args:
        array (numpy.ndarray): The raw 2D input array.
        nodata_val (float): The value representing missing data/oceans.
        fill_value (float, optional): The value to replace NoData with. Defaults to 0.0.

    Returns:
        numpy.ndarray: A normalized 2D array.
    """
    valid_mask = (array != nodata_val) & ~np.isnan(array)
    
    shifted_array = array[valid_mask] - np.min(array[valid_mask]) 
    transformed = np.sqrt(shifted_array)

    min_val = np.min(transformed)
    max_val = np.max(transformed)
    
    processed_array = np.full(array.shape, fill_value, dtype=np.float32)
    normalized_0_1 = (transformed - min_val) / (max_val - min_val)
    processed_array[valid_mask] = (normalized_0_1 * 2.0) - 1.0
    
    return processed_array

def create_terrain_tensor():
    """
    Executes the full pipeline to load, aggregate, clean, and stack WorldClim data.

    Returns:
        numpy.ndarray: A 3D array of shape [3, Height, Width] representing 
        Elevation, Precipitation, and Temperature.
    """
    precip_pattern = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_prec_*.tif")
    temp_pattern   = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_tavg_*.tif")
    elev_pattern   = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_elev.tif")
    
    precip_raw, p_nodata = aggregate_monthly_data(precip_pattern, operation='sum')
    temp_raw, t_nodata   = aggregate_monthly_data(temp_pattern, operation='mean')
    
    elev_file = glob.glob(elev_pattern)[0]
    elev_raw, e_nodata = load_geotiff(elev_file)
    
    assert precip_raw.shape == temp_raw.shape == elev_raw.shape, "Error: Raster shapes do not match"
    
    precip_clean = clean_and_normalize(precip_raw, p_nodata)
    temp_clean   = clean_and_normalize(temp_raw, t_nodata)
    elev_clean   = clean_and_normalize(elev_raw, e_nodata)
    
    stacked_numpy = np.stack([elev_clean, precip_clean, temp_clean], axis=0)
    print(f"Final Tensor Shape: {stacked_numpy.shape}")

    precip_clean = clean_and_normalize(precip_raw, p_nodata)
    del precip_raw  

    temp_clean = clean_and_normalize(temp_raw, t_nodata)
    del temp_raw    

    elev_clean = clean_and_normalize(elev_raw, e_nodata)
    del elev_raw    
    
    del elev_clean, precip_clean, temp_clean 

    return stacked_numpy

if __name__ == "__main__":
    print(f"Starting preprocessing pipeline for {RESOLUTION} resolution...")
    
    dataset_array = create_terrain_tensor()
    
    save_filename = f"worldclim_{RESOLUTION}_full.npy" 
    save_path = PROCESSED_DATA_DIR / save_filename
    
    np.save(save_path, dataset_array)
    
    print("Preprocessing complete!")