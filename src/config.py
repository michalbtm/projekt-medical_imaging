import json


def load_settings():
    """Wczytuje ustawienia z pliku JSON"""
    with open('settings.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_settings(settings):
    """Waliduje ustawienia i sprawdza czy wybrano tylko jeden algorytm odszumiania"""
    denoising = settings['denoising']
    enabled_count = sum([
        denoising['bilateral']['enabled'],
        denoising['tv_chambolle']['enabled'],
        denoising['wavelet']['enabled']
    ])
    
    if enabled_count > 1:
        raise ValueError("[BLAD] Mozna wlaczyc tylko JEDEN algorytm odszumiania!")
    
    # Sprawdzenie czy wybrano przynajmniej jeden algorytm wykrywania krawędzi
    edge_enabled = any(settings['edge_detection'][algo]['enabled'] 
                      for algo in settings['edge_detection'])
    if not edge_enabled:
        raise ValueError("[BLAD] Nalezy wlaczyc przynajmniej JEDEN algorytm wykrywania krawedzi!")
    
    return enabled_count > 0


def display_settings(settings, use_denoising):
    """Wyświetla aktywne ustawienia"""
    # Wyświetlenie aktywnych ustawień
    if use_denoising:
        denoising = settings['denoising']
        for method, config in denoising.items():
            if config['enabled']:
                print(f"\nOdszumianie: {method.upper()}")
                params = {k: v for k, v in config.items() if k != 'enabled'}
                for param, value in params.items():
                    print(f"   - {param}: {value}")
    
    print(f"\nAlgorytmy wykrywania krawedzi:")
    edge_detection = settings['edge_detection']
    for algo, config in edge_detection.items():
        if config['enabled']:
            print(f"   [+] {algo.upper()}")
            params = {k: v for k, v in config.items() if k != 'enabled'}
            for param, value in params.items():
                print(f"       - {param}: {value}")