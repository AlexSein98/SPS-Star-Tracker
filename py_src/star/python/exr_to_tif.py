from PIL import Image
import tifffile
from tifffile import imwrite

import OpenEXR
import Imath

import numpy as np
from matplotlib import pyplot as plt


def exr_to_numpy(filepath: str, channel_name: str = 'Z') -> np.ndarray:
    """
    Reads a single channel from an OpenEXR file and returns it as a NumPy array.

    Args:
        filepath (str): The path to the OpenEXR file.
        channel_name (str): The name of the channel to extract (e.g., 'R', 'G', 'B', 'A', 'Z').

    Returns:
        np.ndarray: A NumPy array containing the pixel data of the specified channel.
    """
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    data_window = header['dataWindow']
    
    # Calculate width and height from the data window
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Determine the pixel type based on the channel's header
    pixel_type = header['channels'][channel_name].type
    
    # Read the raw bytes of the specified channel
    raw_bytes = exr_file.channel(channel_name, pixel_type)
    
    # Convert raw bytes to a NumPy array and reshape it to the image dimensions
    if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        np_dtype = np.float32
    elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
        np_dtype = np.float16
    elif pixel_type == Imath.PixelType(Imath.PixelType.UINT):
        np_dtype = np.uint32
    else:
        raise ValueError(f"Unsupported pixel type: {pixel_type}")

    array_data = np.frombuffer(raw_bytes, dtype=np_dtype).reshape((height, width))
    
    return array_data


thisDir: str = "./py_src/star/"
filePath: str = thisDir + "data/textures/milkyway_2020_4k.exr"


imgArrayR = exr_to_numpy(filePath, 'R')
imgArrayG = exr_to_numpy(filePath, 'G')
imgArrayB = exr_to_numpy(filePath, 'B')

print(f'Max R = {np.max(imgArrayR)}')
print(f'Min R = {np.min(imgArrayR)}')
print(f'Max G = {np.max(imgArrayG)}')
print(f'Min G = {np.min(imgArrayG)}')
print(f'Max B = {np.max(imgArrayB)}')
print(f'Min B = {np.min(imgArrayB)}')

tifffile.imwrite(thisDir + 'python/output/starmap/milkyway_R.tif', imgArrayR.astype(np.float32), compression=0)
tifffile.imwrite(thisDir + 'python/output/starmap/milkyway_G.tif', imgArrayG.astype(np.float32), compression=0)
tifffile.imwrite(thisDir + 'python/output/starmap/milkyway_B.tif', imgArrayB.astype(np.float32), compression=0)
