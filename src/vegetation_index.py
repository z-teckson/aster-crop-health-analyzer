import numpy as np

def calculate_ndvi(nir_band, red_band):
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    Parameters
    ----------
    nir_band : numpy.ndarray
        Near-infrared band array.
    red_band : numpy.ndarray
        Red band array.
    
    Returns
    -------
    numpy.ndarray
        NDVI values in range [-1, 1].
    """
    # Ensure inputs are numpy arrays
    nir = np.asarray(nir_band, dtype=np.float64)
    red = np.asarray(red_band, dtype=np.float64)
    
    numerator = nir - red
    denominator = nir + red
    
    # Avoid division by zero by adding a small epsilon
    # where denominator is zero; set NDVI to 0 in those pixels.
    epsilon = 1e-10
    ndvi = np.where(np.abs(denominator) > epsilon,
                    numerator / denominator,
                    0.0)
    
    # Clip to valid range (optional)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi