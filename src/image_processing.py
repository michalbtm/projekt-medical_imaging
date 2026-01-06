import numpy as np
from skimage import filters, restoration, feature
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte


def denoise_image(img, settings):
    """Stosuje wybrany algorytm odszumiania"""
    denoising = settings['denoising']
    
    # Konwersja do float (0-1)
    img = img.astype(np.float32, copy=False)
    
    # Normalize to [0, 1] range if needed
    if img.max() > 1.0:
        img = img / 255.0

    # Obsługa różnych formatów obrazów
    if img.ndim == 3 and img.shape[2] == 2:
        # Grayscale + Alpha - używamy tylko pierwszego kanału
        img = img[:, :, 0]
    
    if denoising['bilateral']['enabled']:
        params = denoising['bilateral']
        # Dla obrazów RGB dodajemy channel_axis
        if img.ndim == 3:
            return restoration.denoise_bilateral(
                img, 
                sigma_color=params['sigma_color'],
                sigma_spatial=params['sigma_spatial'],
                channel_axis=-1
            )
        else:
            return restoration.denoise_bilateral(
                img, 
                sigma_color=params['sigma_color'],
                sigma_spatial=params['sigma_spatial']
            )
    
    elif denoising['tv_chambolle']['enabled']:
        params = denoising['tv_chambolle']
        return restoration.denoise_tv_chambolle(
            img,
            weight=params['weight'],
            channel_axis=-1 if img.ndim == 3 else None
        )
    
    elif denoising['wavelet']['enabled']:
        params = denoising['wavelet']
        return restoration.denoise_wavelet(
            img,
            sigma=params['sigma'],
            mode=params['mode'],
            wavelet_levels=params['wavelet_levels'],
            channel_axis=-1 if img.ndim == 3 else None
        )
    
    return img


def apply_edge_detection(img, algorithm, params):
    """Stosuje wybrany algorytm wykrywania krawędzi"""
    # Konwersja do grayscale - obsługa różnych formatów
    if img.ndim == 2:
        # Już grayscale
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 1:
            # Grayscale z jednym kanałem
            gray = img[:, :, 0]
        elif img.shape[2] == 2:
            # Grayscale + Alpha - bierzemy tylko grayscale
            gray = img[:, :, 0]
        elif img.shape[2] == 3:
            # RGB
            gray = rgb2gray(img)
        elif img.shape[2] == 4:
            # RGBA - ignorujemy alpha
            gray = rgb2gray(img[:, :, :3])
        else:
            raise ValueError(f"Nieobslugiwany format obrazu: {img.shape}")
    else:
        raise ValueError(f"Nieobslugiwany wymiar obrazu: {img.ndim}")
    
    gray = gray.astype(np.float32, copy=False)
    
    if algorithm == 'canny':
        # Canny może być w filters lub feature
        try:
            canny_func = filters.canny
        except AttributeError:
            canny_func = feature.canny
            
        result = canny_func(
            gray,
            sigma=params['sigma'],
            low_threshold=params['low_threshold'],
            high_threshold=params['high_threshold']
        )
        # Canny zwraca bool, konwertujemy do uint8
        return result.astype(np.uint8) * 255
    
    elif algorithm == 'sobel':
        result = filters.sobel(gray)
        if params['use_threshold']:
            result = (result > params['threshold']).astype(np.uint8) * 255
        else:
            # Normalizacja do 0-255
            result = (result - result.min()) / (result.max() - result.min())
            result = (result * 255).astype(np.uint8)
        return result
    
    elif algorithm == 'prewitt':
        result = filters.prewitt(gray)
        if params['use_threshold']:
            result = (result > params['threshold']).astype(np.uint8) * 255
        else:
            # Normalizacja do 0-255
            result = (result - result.min()) / (result.max() - result.min())
            result = (result * 255).astype(np.uint8)
        return result
    
    return gray