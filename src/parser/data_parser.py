"""
Universal data parser for medical imaging datasets.
Handles both microscopy (regular image formats) and MRI (nii.gz) data.
Provides unified interface for processing pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
from dataclasses import dataclass
import gzip
import shutil


@dataclass
class ImageMetadata:
    """Metadata for processed medical images"""
    source_file: Path
    filename: str
    modality: str  # 'microscopy' or 'mri'
    patient_id: str
    slice_info: Optional[str] = None  # For MRI: t{:02d}_z{:03d}, for microscopy: None


class DataParser:
    """Universal parser for medical imaging data"""
    
    # Supported microscopy formats
    MICROSCOPY_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, data_root: Path = None, include_archive: bool = False):
        """
        Initialize parser.
        
        Args:
            data_root: Root directory of data. Defaults to 'data/Resources'
            include_archive: If True, also parse archive/unmodified-data
        """
        if data_root is None:
            data_root = Path('data') / 'Resources'
        
        self.data_root = Path(data_root)
        self.include_archive = include_archive
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        self.archive_root = Path('data') / 'microscopy' if include_archive else None
        if self.include_archive and not self.archive_root.exists():
            print(f"[WARNING] Microscopy directory not found: {self.archive_root}")
            self.archive_root = None
    
    def get_datasets(self) -> dict:
        """
        Get available datasets (training/testing).
        
        Returns:
            dict: {dataset_name: Path}
        """
        datasets = {}
        for dataset_dir in self.data_root.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name in ['training', 'testing']:
                datasets[dataset_dir.name] = dataset_dir
        
        if self.archive_root:
            archive_map = {'train': 'train_archive', 'test': 'test_archive'}
            for archive_dir_name, dataset_name in archive_map.items():
                archive_dir = self.archive_root / 'unmodified-data' / archive_dir_name
                if archive_dir.exists():
                    datasets[dataset_name] = archive_dir
        
        return datasets
    
    def get_patients(self, dataset_name: str = 'training') -> list:
        """
        Get all patient directories in a dataset.
        
        Args:
            dataset_name: 'training', 'testing', 'train_archive', or 'test_archive'
        
        Returns:
            list: Sorted list of patient directory paths, or empty list for archive
        """
        # Archive doesn't have patient structure
        if dataset_name in ['train_archive', 'test_archive']:
            return []
        
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            return []
        
        patients = [p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith('patient')]
        return sorted(patients)
    
    def parse_microscopy_image(self, image_path: Path) -> Generator[Tuple[np.ndarray, ImageMetadata], None, None]:
        """
        Parse a single microscopy image file.
        
        Args:
            image_path: Path to microscopy image file
        
        Yields:
            (image_array, metadata) tuples
        """
        try:
            from skimage import io
            
            if image_path.suffix.lower() not in self.MICROSCOPY_EXTENSIONS:
                return
            
            img = io.imread(image_path)
            
            # Extract patient ID from parent directory if possible
            parent_dir = image_path.parent
            patient_id = parent_dir.name if parent_dir.name.startswith('patient') else 'unknown'
            
            metadata = ImageMetadata(
                source_file=image_path,
                filename=image_path.name,
                modality='microscopy',
                patient_id=patient_id
            )
            
            yield img, metadata
            
        except Exception as e:
            print(f"[ERROR] Failed to load microscopy image {image_path}: {e}")
    
    def parse_mri_image(self, mri_path: Path) -> Generator[Tuple[np.ndarray, ImageMetadata], None, None]:
        """
        Parse MRI nii.gz file and yield 2D slices.
        
        Args:
            mri_path: Path to .nii.gz MRI file
        
        Yields:
            (slice_2d, metadata) tuples
        """
        # Only process .nii.gz files
        if mri_path.suffix != '.gz' or not mri_path.stem.endswith('.nii'):
            return
        
        try:
            import nibabel as nib
            from skimage.exposure import rescale_intensity
            
            parent_dir = mri_path.parent
            patient_id = parent_dir.name if parent_dir.name.startswith('patient') else 'unknown'
            
            nii = nib.load(str(mri_path))
            data = nii.get_fdata()
            
            # Handle both 3D and 4D MRI files
            if data.ndim == 3:
                # 3D file: X, Y, Z
                X, Y, Z = data.shape
                for z in range(Z):
                    slice_2d = data[:, :, z]
                    slice_2d = rescale_intensity(
                        slice_2d,
                        in_range='image',
                        out_range=(0.0, 1.0)
                    ).astype(np.float32)
                    
                    slice_name = f"{mri_path.stem}_z{z:03d}"
                    metadata = ImageMetadata(
                        source_file=mri_path,
                        filename=f"{slice_name}",
                        modality='mri',
                        patient_id=patient_id,
                        slice_info=slice_name
                    )
                    yield slice_2d, metadata
                    
            elif data.ndim == 4:
                # 4D file: X, Y, Z, T (time/frames)
                X, Y, Z, T = data.shape
                for t in range(T):
                    for z in range(Z):
                        slice_2d = data[:, :, z, t]
                        slice_2d = rescale_intensity(
                            slice_2d,
                            in_range='image',
                            out_range=(0.0, 1.0)
                        ).astype(np.float32)
                        
                        slice_name = f"{mri_path.stem}_t{t:02d}_z{z:03d}"
                        metadata = ImageMetadata(
                            source_file=mri_path,
                            filename=f"{slice_name}",
                            modality='mri',
                            patient_id=patient_id,
                            slice_info=slice_name
                        )
                        yield slice_2d, metadata
                
        except Exception as e:
            print(f"[ERROR] Failed to load MRI file {mri_path}: {e}")
    
    def parse_patient_directory(self, patient_dir: Path) -> Generator[Tuple[np.ndarray, ImageMetadata], None, None]:
        """
        Parse all images in a patient directory (both microscopy and MRI).
        
        Args:
            patient_dir: Path to patient directory
        
        Yields:
            (image_array, metadata) tuples
        """
        if not patient_dir.is_dir():
            return
        
        # Look for microscopy images
        for image_file in patient_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in self.MICROSCOPY_EXTENSIONS:
                yield from self.parse_microscopy_image(image_file)
        
        # Look for MRI files
        for mri_file in patient_dir.glob('*.nii.gz'):
            # Skip ground truth files
            if '_gt' in mri_file.name:
                continue
            
            yield from self.parse_mri_image(mri_file)
    
    def parse_dataset(self, dataset_name: str = 'training') -> Generator[Tuple[np.ndarray, ImageMetadata], None, None]:
        """
        Parse entire dataset (all patients) or archive.
        
        Args:
            dataset_name: 'training', 'testing', 'train_archive', or 'test_archive'
        
        Yields:
            (image_array, metadata) tuples
        """
        # Check if this is an archive dataset
        if dataset_name in ['train_archive', 'test_archive']:
            archive_subdir = 'train' if dataset_name == 'train_archive' else 'test'
            if not self.archive_root:
                return
            
            archive_dataset_path = self.archive_root / 'unmodified-data' / archive_subdir
            if not archive_dataset_path.exists():
                return
            
            # Parse archive flat structure (imgs/ and labels/ folders)
            imgs_dir = archive_dataset_path / 'imgs'
            if not imgs_dir.exists():
                return
            
            for image_file in sorted(imgs_dir.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() in self.MICROSCOPY_EXTENSIONS:
                    yield from self.parse_microscopy_image(image_file)
        else:
            # Parse normal structured dataset
            patients = self.get_patients(dataset_name)
            
            for patient_dir in patients:
                yield from self.parse_patient_directory(patient_dir)
    
    def parse_all(self) -> Generator[Tuple[np.ndarray, ImageMetadata], None, None]:
        """
        Parse both training and testing datasets.
        
        Yields:
            (image_array, metadata) tuples
        """
        for dataset_name in ['training', 'testing']:
            yield from self.parse_dataset(dataset_name)
    
    @staticmethod
    def get_dataset_statistics(parser_generator: Generator) -> dict:
        """
        Get statistics from a parser generator without consuming it.
        This is mainly informational - iterates through all data.
        
        Args:
            parser_generator: Generator from parse_* methods
        
        Returns:
            dict: Statistics about the dataset
        """
        stats = {
            'total_images': 0,
            'by_modality': {'microscopy': 0, 'mri': 0},
            'by_patient': {},
            'unique_patients': set()
        }
        
        for image, metadata in parser_generator:
            stats['total_images'] += 1
            stats['by_modality'][metadata.modality] += 1
            stats['unique_patients'].add(metadata.patient_id)
            
            if metadata.patient_id not in stats['by_patient']:
                stats['by_patient'][metadata.patient_id] = 0
            stats['by_patient'][metadata.patient_id] += 1
        
        stats['unique_patients'] = len(stats['unique_patients'])
        return stats


def create_symlinks_to_input(source_dataset: str = 'training', 
                             input_dir: Path = None,
                             data_root: Path = None) -> int:
    """
    Create symlinks in 'input' directory pointing to all images in specified dataset.
    Useful for using existing processing pipeline with structured data.
    
    Args:
        source_dataset: 'training' or 'testing'
        input_dir: Target input directory. Defaults to 'input'
        data_root: Data root directory. Defaults to 'data/Resources'
    
    Returns:
        int: Number of symlinks created
    """
    if input_dir is None:
        input_dir = Path('input')
    if data_root is None:
        data_root = Path('data') / 'Resources'
    
    input_dir = Path(input_dir)
    input_dir.mkdir(exist_ok=True)
    
    parser = DataParser(data_root)
    count = 0
    
    for image_array, metadata in parser.parse_dataset(source_dataset):
        # Create output filename
        if metadata.slice_info:
            output_name = f"{metadata.patient_id}_{metadata.slice_info}.png"
        else:
            output_name = f"{metadata.patient_id}_{metadata.filename}"
        
        symlink_path = input_dir / output_name
        
        try:
            # Convert nii.gz to PNG if needed
            if metadata.modality == 'mri':
                from skimage import io
                from skimage.util import img_as_ubyte
                
                # Normalize to 0-255 range
                if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                    img_uint8 = img_as_ubyte(image_array)
                else:
                    img_uint8 = image_array
                
                io.imsave(symlink_path, img_uint8, check_contrast=False)
                print(f"[OK] Saved MRI slice: {output_name}")
            else:
                # For microscopy, create symlink
                if symlink_path.exists():
                    symlink_path.unlink()
                symlink_path.symlink_to(metadata.source_file.absolute())
                print(f"[OK] Symlinked: {output_name}")
            
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to create link for {output_name}: {e}")
    
    return count


def reconstruct_mri_nii(processed_slices_dir: Path, output_dir: Path, patient_ids: list = None) -> int:
    """
    Reconstruct 4D NII.GZ files from processed 2D slices.
    
    Args:
        processed_slices_dir: Directory containing processed slices (from output/)
        output_dir: Where to save reconstructed NII.GZ files
        patient_ids: If specified, only reconstruct these patients
    
    Returns:
        int: Number of reconstructed files
    """
    import nibabel as nib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import natsort
        sort_func = natsort.natsorted
    except ImportError:
        sort_func = sorted
    
    reconstructed = 0
    
    # Get all PNG files
    png_files = list(processed_slices_dir.glob('**/*.png'))
    
    # Group by original MRI file
    groups = {}
    for png_file in png_files:
        # Parse filename: patientXXX_[originalname]_t##_z###.png
        stem = png_file.stem
        parts = stem.split('_')
        
        if len(parts) >= 4 and parts[-2].startswith('t') and parts[-1].startswith('z'):
            patient_id = parts[0]
            original_name = '_'.join(parts[1:-2])
            t_idx = int(parts[-2][1:])
            z_idx = int(parts[-1][1:])
            
            if patient_ids and patient_id not in patient_ids:
                continue
            
            key = (patient_id, original_name)
            if key not in groups:
                groups[key] = {}
            
            groups[key][(t_idx, z_idx)] = png_file
    
    # Reconstruct each group
    from skimage import io
    
    for (patient_id, original_name), slices_dict in groups.items():
        if not slices_dict:
            continue
        
        # Sort slices
        sorted_slices = sort_func(slices_dict.items())
        
        # Get dimensions from first slice
        first_img = io.imread(sorted_slices[0][1])
        X, Y = first_img.shape[:2]
        
        # Get unique t and z indices
        t_indices = sorted(set(t_idx for (t_idx, z_idx) in slices_dict.keys()))
        z_indices = sorted(set(z_idx for (t_idx, z_idx) in slices_dict.keys()))
        
        Z = len(z_indices)
        T = len(t_indices)
        
        # Create 4D array
        data_4d = np.zeros((X, Y, Z, T), dtype=np.float32)
        
        # Fill array
        for (t_idx, z_idx), slice_path in slices_dict.items():
            img = io.imread(slice_path)
            if img.ndim == 3:
                img = img[:, :, 0]
            
            # Normalize to [0, 1]
            if img.max() > 1:
                img = img / 255.0
            
            t_pos = t_indices.index(t_idx)
            z_pos = z_indices.index(z_idx)
            data_4d[:, :, z_pos, t_pos] = img
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(data_4d, np.eye(4))
        
        # Save
        output_path = output_dir / f"{patient_id}_{original_name}_reconstructed.nii.gz"
        nib.save(nii_img, output_path)
        print(f"[OK] Reconstructed: {output_path.name}")
        reconstructed += 1
    
    return reconstructed


if __name__ == '__main__':
    # Example usage
    parser = DataParser()
    
    print("=" * 60)
    print("Medical Imaging Data Parser - Example Usage")
    print("=" * 60)
    
    # List available datasets
    datasets = parser.get_datasets()
    print(f"\nAvailable datasets: {list(datasets.keys())}")
    
    # Get training patients
    training_patients = parser.get_patients('training')
    print(f"\nTraining patients found: {len(training_patients)}")
    if training_patients:
        print(f"  First patient: {training_patients[0].name}")
    
    # Parse first patient and show statistics
    if training_patients:
        print(f"\nParsing first patient...")
        count = 0
        for image, metadata in parser.parse_patient_directory(training_patients[0]):
            count += 1
            if count == 1:
                print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
                print(f"  Modality: {metadata.modality}, Patient: {metadata.patient_id}")
        
        print(f"  Total images/slices in patient: {count}")
    
    # Quick stats
    print("\n" + "=" * 60)
    print("To use with existing pipeline:")
    print("  1. Call create_symlinks_to_input('training') - creates input/")
    print("  2. Run main.py as usual")
    print("=" * 60)
