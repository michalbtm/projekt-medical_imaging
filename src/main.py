import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from config import load_settings, validate_settings, display_settings
from processor import process_single_image, get_image_files, clean_output_directory


def main():
    """Główna funkcja programu"""
    try:
        # Wczytanie i walidacja ustawień
        settings = load_settings()
        use_denoising = validate_settings(settings)
        
    except ValueError as e:
        print(f"\n{e}\n")
        return
    except FileNotFoundError:
        print("\n[BLAD] Nie znaleziono pliku 'settings.json'\n")
        return
    
    # Przygotowanie katalogów
    input_dir = Path('input')
    output_dir = Path('output')
    
    if not input_dir.exists():
        print("\n[BLAD] Katalog 'input' nie istnieje!\n")
        return
    
    # Czyszczenie katalogu output
    clean_output_directory(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Znalezienie wszystkich obrazów
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print("\n[BLAD] Nie znaleziono obrazow w katalogu 'input'\n")
        return
    
    print(f"\n{'='*50}")
    print(f"Znaleziono {len(image_files)} obrazow do przetworzenia")
    print(f"{'='*50}")
    
    # Wyświetlenie konfiguracji
    display_settings(settings, use_denoising)
    
    print(f"\n{'='*50}")
    print("Rozpoczynam przetwarzanie...\n")
    
    # Przetwarzanie wielowątkowe
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_single_image, img, settings, output_dir, use_denoising)
            for img in image_files
        ]
        
        for future in futures:
            if future.result():
                success_count += 1
            else:
                fail_count += 1
    
    print(f"\n{'='*50}")
    print(f"[OK] GOTOWE!")
    print(f"   Przetworzone: {success_count}/{len(image_files)}")
    if fail_count > 0:
        print(f"   Bledy: {fail_count}")
    print(f"   Wyniki w katalogu 'output'")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()