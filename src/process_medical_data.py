"""
Integration script: Process medical imaging data using the existing pipeline.
Uses processor.py's architecture with data_parser to handle structured medical data.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_settings, validate_settings, display_settings
from processor import process_algorithm, clean_output_directory
from parser.data_parser import DataParser, ImageMetadata
from skimage import io
from skimage.util import img_as_ubyte


def normalize_to_uint8(img_array):
    """Normalizuj obraz do zakresu uint8."""
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        if img_array.max() <= 1.0:
            return (img_array * 255).astype(np.uint8)
        else:
            return img_as_ubyte(img_array)
    return img_array.astype(np.uint8)


def parse_range(range_str):
    """Parse range string like '001-003' or '3-6' into list of values.
    
    Examples:
        '5' → [5]
        '001' → ['001']
        '001-003' → ['001', '002', '003']
        '3-6' → [3, 4, 5, 6]
    """
    # Check if single value with padding (e.g., '001')
    if '-' not in range_str:
        if range_str[0] == '0' and len(range_str) > 1:
            return [range_str]  # Keep as string if padded
        try:
            return [int(range_str)]
        except ValueError:
            return [range_str]
    
    # Handle range (e.g., '001-003' or '0-2')
    parts = range_str.split('-')
    start_str, end_str = parts[0], parts[-1]
    is_padded = len(start_str) > 1 and start_str[0] == '0'
    
    try:
        start = int(start_str)
        end = int(end_str)
        
        if is_padded:
            width = len(start_str)
            return [str(i).zfill(width) for i in range(start, end + 1)]
        else:
            return list(range(start, end + 1))
    except ValueError:
        return [range_str]


def process_selective(mri_patients=None, mri_z_range=None, mri_t_range=None,
                     microscopy_ids=None, output_dir=None, data_root=None):
    """Process selected MRI patients/frames/slices and/or microscopy samples.
    
    Args:
        mri_patients: List of patient IDs (e.g., ['001', '002'] or None for all)
        mri_z_range: List of z-slice indices to process (e.g., [0,1,2] or None for all)
        mri_t_range: List of frame indices to process (e.g., [0,1] or None for all)
        microscopy_ids: List of microscopy sample IDs or None for none
        output_dir: Output directory
        data_root: MRI data root directory
    """
    if output_dir is None:
        output_dir = Path('output')
    if data_root is None:
        data_root = Path('data') / 'mri'
    
    try:
        settings = load_settings()
        use_denoising = validate_settings(settings)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}\n")
        return
    
    output_dir = Path(output_dir)
    clean_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        parser = DataParser(data_root, include_archive=False)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        return
    
    all_images = []
    processed_count = 0
    
    # Process MRI patients if specified
    if mri_patients:
        print(f"Przetwarzanie pacjentów MRI: {mri_patients}")
        if mri_z_range:
            print(f"  Z-slices: {mri_z_range}")
        if mri_t_range:
            print(f"  Frames: {mri_t_range}")
        
        training_patients = parser.get_patients('training')
        
        for patient_dir in training_patients:
            patient_id = patient_dir.name.replace('patient', '')  # Get just the number part
            if patient_id not in mri_patients:
                continue
            
            for img_array, metadata in parser.parse_patient_directory(patient_dir):
                if metadata.modality != 'mri':
                    continue
                
                # Parse slice info: format is "patientXXX_4d.nii_t##_z###"
                if metadata.slice_info:
                    parts = metadata.slice_info.split('_')
                    t_idx = int(parts[-2][1:]) if parts[-2].startswith('t') else None
                    z_idx = int(parts[-1][1:]) if parts[-1].startswith('z') else None
                    
                    # Filter by t and z ranges
                    if mri_t_range and t_idx not in mri_t_range:
                        continue
                    if mri_z_range and z_idx not in mri_z_range:
                        continue
                
                all_images.append((img_array, metadata))
    
    # Process microscopy if specified
    if microscopy_ids:
        print(f"Przetwarzanie microscopy samples: {microscopy_ids}")
        microscopy_root = Path('data') / 'microscopy' / 'unmodified-data'
        
        # Convert all IDs to 4-digit strings for uniform matching.
        # This handles '1' -> '0001', '1-5' -> ['0001', ..., '0005'], and '001' -> '0001'
        formatted_ids = []
        for i in microscopy_ids:
            if isinstance(i, int):
                formatted_ids.append(f"{i:04d}")
            elif isinstance(i, str) and i.isdigit():
                formatted_ids.append(i.zfill(4))
            else:
                formatted_ids.append(i) # Keep as is if not a simple number

        for dataset_type in ['train', 'test']:
            dataset_path = microscopy_root / dataset_type / 'imgs'
            if not dataset_path.is_dir():
                continue

            for img_path in dataset_path.glob('frame_*.png'):
                try:
                    # Extract ID from filename like 'frame_0001.png' -> '0001'
                    file_id = img_path.stem.split('_')[-1]
                    
                    if file_id in formatted_ids:
                        img_array = io.imread(img_path)
                        metadata = ImageMetadata(
                            source_file=img_path,
                            filename=img_path.name,
                            modality='microscopy',
                            patient_id=f"{dataset_type}_{file_id}"
                        )
                        all_images.append((img_array, metadata))
                        print(f"  Znaleziono i dodano: {img_path}")
                except Exception as e:
                    print(f"[ERROR] Błąd podczas przetwarzania {img_path}: {e}")
    
    if not all_images:
        print("\n[ERROR] Brak obrazów spełniających kryteria\n")
        return
    
    print(f"Przetwarzanie {len(all_images)} obrazów\n")
    display_settings(settings, use_denoising)
    
    success_count = 0
    
    def process_image_wrapper(image_array, metadata):
        try:
            img = normalize_to_uint8(image_array.copy())
            
            from image_processing import denoise_image
            if use_denoising:
                img = denoise_image(img, settings)
                img = normalize_to_uint8(img)
            
            if metadata.slice_info:
                img_name = f"{metadata.patient_id}_{metadata.slice_info}"
            else:
                img_name = f"{metadata.patient_id}_{Path(metadata.filename).stem}"
            
            edge_detection = settings['edge_detection']
            enabled_algos = [(name, config) for name, config in edge_detection.items() 
                            if config['enabled']]
            
            with ThreadPoolExecutor(max_workers=len(enabled_algos)) as executor:
                futures = [
                    executor.submit(process_algorithm, img, img_name, algo_name, 
                                  algo_config, output_dir)
                    for algo_name, algo_config in enabled_algos
                ]
                results = [f.result() for f in futures]
            
            if all(results):
                print(f"[OK] {img_name}")
                return True
            else:
                print(f"[OK] {img_name}")
                return True
                
        except Exception as e:
            print(f"[ERROR] {metadata.filename}: {e}")
            return False
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_image_wrapper, img_array, metadata)
            for img_array, metadata in all_images
        ]
        
        for future in futures:
            if future.result():
                success_count += 1
    
    print(f"\nGotowe! Przetworzono: {success_count}/{len(all_images)}")
    print(f"Wyniki: '{output_dir}'\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process medical imaging data with selective filtering'
    )
    parser.add_argument(
        'command',
        choices=['process', 'info'],
        help='Polecenie do wykonania'
    )
    parser.add_argument(
        '--mri',
        nargs=3,
        metavar=('PATIENTS', 'Z_SLICES', 'FRAMES'),
        help='Process MRI: patient IDs (e.g. 001 or 001-003), z-slices (e.g. 0-5), frames (e.g. 0-2)'
    )
    parser.add_argument(
        '--microscopy',
        metavar='IDS',
        help='Process microscopy: sample IDs (e.g. 1 or 1-10)'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data') / 'mri',
        help='Path to MRI data root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Path to output directory'
    )
    
    args = parser.parse_args()
    
    if args.command == 'process':
        if not args.mri and not args.microscopy:
            print("[ERROR] Podaj --mri i/lub --microscopy")
            print("Przykład: python src/process_medical_data.py process --mri 001 0-5 0-2")
            print("         python src/process_medical_data.py process --microscopy 1-10")
        else:
            mri_patients = None
            mri_z_range = None
            mri_t_range = None
            
            if args.mri:
                patients_str, z_str, t_str = args.mri
                mri_patients = parse_range(patients_str)
                mri_z_range = parse_range(z_str)
                mri_t_range = parse_range(t_str)
            
            microscopy_ids = None
            if args.microscopy:
                microscopy_ids = parse_range(args.microscopy)
            
            process_selective(
                mri_patients=mri_patients,
                mri_z_range=mri_z_range,
                mri_t_range=mri_t_range,
                microscopy_ids=microscopy_ids,
                output_dir=args.output_dir,
                data_root=args.data_root
            )
    
    elif args.command == 'info':
        try:
            parser_obj = DataParser(args.data_root, include_archive=False)
            datasets = parser_obj.get_datasets()
            
            print(f"\nZbiory danych:")
            print(f"Katalog: {args.data_root}")
            print(f"Dostępne: {list(datasets.keys())}\n")
            
            for dataset_name in datasets:
                patients = parser_obj.get_patients(dataset_name)
                print(f"{dataset_name.upper()}:")
                print(f"  Pacjenci: {len(patients)}")
                
                if patients:
                    image_count = sum(1 for _ in parser_obj.parse_patient_directory(patients[0]))
                    print(f"  Przykład ({patients[0].name}): {image_count} obrazów")
        
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
