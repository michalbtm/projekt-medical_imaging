import numpy as np
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage import io
from skimage.util import img_as_ubyte

from image_processing import denoise_image, apply_edge_detection


def clean_output_directory(output_dir):
    """Usuwa katalog output jeśli istnieje"""
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[INFO] Wyczyszczono katalog '{output_dir}'\n")


def process_algorithm(img, img_name, algo_name, algo_config, output_base):
    """Przetwarza obraz pojedynczym algorytmem"""
    try:
        result = apply_edge_detection(img, algo_name, algo_config)
        
        # Tworzenie katalogu dla algorytmu
        algo_dir = output_base / algo_name
        algo_dir.mkdir(parents=True, exist_ok=True)
        
        # Zapis wyniku - wymuszamy PNG
        output_path = algo_dir / f"{img_name}.png"
        io.imsave(output_path, result, check_contrast=False)
        return True
    except Exception as e:
        print(f"[BLAD] {algo_name} dla {img_name}: {e}")
        return False


def process_single_image(img_path, settings, output_base, use_denoising):
    """Przetwarza jeden obraz według ustawień"""
    img_name = Path(img_path)
    try:

        img = io.imread(img_path)
        
        # Odszumianie jeśli włączone
        if use_denoising:
            img = denoise_image(img, settings)
            # Po denoisingu konwertujemy z powrotem do uint8 dla edge detection
            if img.dtype == np.float64 or img.dtype == np.float32:
                img = img_as_ubyte(img)
        
        # Przetwarzanie równoległe wszystkich algorytmów dla tego obrazu
        edge_detection = settings['edge_detection']
        enabled_algos = [(name, config) for name, config in edge_detection.items() if config['enabled']]
        
        with ThreadPoolExecutor(max_workers=len(enabled_algos)) as executor:
            futures = [
                executor.submit(process_algorithm, img, img_name, algo_name, algo_config, output_base)
                for algo_name, algo_config in enabled_algos
            ]
            results = [f.result() for f in futures]
        
        if all(results):
            print(f"[OK] Przetworzono: {img_name}")
            return True
        else:
            print(f"[CZESC] Czesciowo przetworzono: {img_name}")
            return True
        
    except Exception as e:
        print(f"[BLAD] Przy {img_name}: {e}")
        return False


def get_image_files(input_dir):
    """Znajduje wszystkie obrazy w katalogu input"""
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    # Sortowanie dla lepszej czytelności
    return sorted(image_files)