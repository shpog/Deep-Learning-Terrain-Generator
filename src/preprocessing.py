import os
import glob
import rasterio
import numpy as np
import torch

from config import RESOLUTION, RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_geotiff(file_path):
    """
    Loads a GeoTIFF file and extracts its first band along with its nodata value.

    :param file_path: The file path to the GeoTIFF raster.
    :type file_path: str or pathlib.Path
    :return: A tuple containing the image data as a 2D array and the raster's nodata value.
    :rtype: tuple(numpy.ndarray, float or int)
    """
    with rasterio.open(file_path) as src:
        return src.read(1), src.nodata


def aggregate_monthly_data(file_pattern, operation="sum"):
    """
    Aggregates multiple GeoTIFF files matching a specific pattern using a mathematical operation.

    This function sequentially reads multiple rasters (e.g., monthly climate data),
    masks out nodata values, and calculates either the pixel-wise sum or mean across all files
    in the sequence.

    :param file_pattern: A glob pattern used to locate the files for aggregation.
    :type file_pattern: str
    :param operation: The aggregation operation to perform. Accepted values are 'sum' or 'mean'. Defaults to 'sum'.
    :type operation: str, optional
    :raises FileNotFoundError: If no files are found matching the provided ``file_pattern``.
    :return: A tuple containing the aggregated 2D data array and the original nodata value.
    :rtype: tuple(numpy.ndarray, float or int)
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")

    print(f"Aggregating {len(files)} files from {file_pattern} using '{operation}'...")

    first_arr, nodata_val = load_geotiff(files[0])
    valid_mask = first_arr != nodata_val

    accumulator = np.zeros(first_arr.shape, dtype=np.float32)
    accumulator[valid_mask] = first_arr[valid_mask]

    del first_arr

    for f in files[1:]:
        arr, _ = load_geotiff(f)
        accumulator[valid_mask] += arr[valid_mask]
        del arr

    if operation == "mean":
        accumulator[valid_mask] /= len(files)

    aggregated = np.full(accumulator.shape, nodata_val, dtype=np.float32)
    aggregated[valid_mask] = accumulator[valid_mask]

    return aggregated, nodata_val


def clean_and_normalize(array, nodata_val, fill_value=0.0):
    """
    Cleans and normalizes a numerical array to a standardized range of [-1.0, 1.0].

    This process identifies valid pixels by filtering out nodata and NaN values. It then shifts
    the data to a baseline of zero, applies a square root transformation to handle skewed
    distributions, and normalizes the valid data into the [-1.0, 1.0] range. Invalid pixels
    are replaced with the designated fill value.

    :param array: The input 2D raster array to be processed.
    :type array: numpy.ndarray
    :param nodata_val: The specific value representing missing or nodata pixels in the input array.
    :type nodata_val: float or int
    :param fill_value: The static value assigned to invalid or nodata pixels in the final output. Defaults to 0.0.
    :type fill_value: float, optional
    :return: The processed array normalized to the [-1.0, 1.0] range.
    :rtype: numpy.ndarray
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
    Orchestrates the loading, aggregation, and normalization of raw terrain and climate datasets.

    This pipeline utilizes paths defined in the project's configuration to aggregate monthly
    precipitation (using summation) and temperature (using average) data. It then cleans,
    normalizes, and stacks the elevation, precipitation, and temperature arrays into a
    single multi-channel tensor. Intermediate memory is actively freed during execution.

    :raises AssertionError: If the extracted rasters for precipitation, temperature, and elevation do not share the same dimensions.
    :return: A stacked multi-channel array of shape (3, Height, Width) representing normalized elevation, precipitation, and temperature.
    :rtype: numpy.ndarray
    """
    precip_pattern = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_prec_*.tif")
    temp_pattern = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_tavg_*.tif")
    elev_pattern = str(RAW_DATA_DIR / f"wc2.1_{RESOLUTION}_elev.tif")

    precip_raw, p_nodata = aggregate_monthly_data(precip_pattern, operation="sum")
    temp_raw, t_nodata = aggregate_monthly_data(temp_pattern, operation="mean")

    elev_file = glob.glob(elev_pattern)[0]
    elev_raw, e_nodata = load_geotiff(elev_file)

    assert (
        precip_raw.shape == temp_raw.shape == elev_raw.shape
    ), "Error: Raster shapes do not match"

    precip_clean = clean_and_normalize(precip_raw, p_nodata)
    temp_clean = clean_and_normalize(temp_raw, t_nodata)
    elev_clean = clean_and_normalize(elev_raw, e_nodata)

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
