import os
import glob
import rasterio
import numpy as np
import torch

from config import RESOLUTION, RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_geotiff(file_path):
    """Reads a single GeoTIFF."""
    with rasterio.open(file_path) as src:
        return src.read(1), src.nodata

def aggregate_monthly_data(file_pattern, operation='sum'):
    """
    Finds all 12 monthly files matching the pattern and aggregates them iteratively
    to prevent RAM overflow on high-resolution (e.g., 30s) datasets.
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
    """Isolates valid data, normalizes to [0, 1], and fills NoData gaps."""
    valid_mask = (array != nodata_val) & ~np.isnan(array)
    
    min_val = np.min(array[valid_mask])
    max_val = np.max(array[valid_mask])
    
    processed_array = np.full(array.shape, fill_value, dtype=np.float32)
    processed_array[valid_mask] = (array[valid_mask] - min_val) / (max_val - min_val)
    
    return processed_array

def create_terrain_tensor():
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