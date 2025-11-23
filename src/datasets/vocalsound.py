from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import torch.nn as nn
import torch


class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class VocalSound(AudioDataset):
    base_folder = 'VocalSound'
    url = "https://github.com/YuanGongND/vocalsound/archive/master.zip"  # URL da verificare
    filename = "vocalsound.zip"
    audio_extensions = ['.wav', '.WAV']
    label_col = 'category'
    file_col = 'filename'
    
    # Categorie vocali del dataset
    vocal_categories = ['cough', 'laughter', 'sigh', 'sneeze', 'sniff', 'throatclearing']
    
    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True, validate: bool = True):
        super().__init__(root)
        self._load_meta()
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        
        print("Loading audio files")
        
        # Processa il DataFrame
        self.df['category'] = self.df['category'].str.replace('_', ' ')
        
        for _, row in tqdm(self.df.iterrows()):
            file_path = row[self.file_col]
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)
        
        # Valida i file audio
        if validate:
            self.validate_audio_files()
    
    def _load_meta(self):
        """Crea il DataFrame esplorando la struttura delle cartelle audio"""
        base_path = os.path.join(self.root, self.base_folder)
        
        # Liste per creare il DataFrame
        data = {
            'filename': [],
            'category': [],
            'speaker_id': []
        }
        
        # Crea il mapping classe-indice basato sulle categorie vocali
        self.classes = sorted(self.vocal_categories)
        self.class_to_idx = {}
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i
        
        # Cerca tutti i file audio nella cartella base
        if os.path.isdir(base_path):
            for audio_file in sorted(os.listdir(base_path)):
                if any(audio_file.endswith(ext) for ext in self.audio_extensions):
                    # Estrae la categoria dal nome del file
                    # Formato: speakerID_index_category.wav (es. m0246_0_laughter.wav)
                    parts = audio_file.rsplit('_', 1)
                    if len(parts) == 2:
                        category = parts[1].replace('.wav', '').replace('.WAV', '')
                        
                        if category in self.vocal_categories:
                            # Estrae speaker ID
                            speaker_parts = audio_file.split('_')
                            speaker_id = speaker_parts[0] if speaker_parts else 'unknown'
                            
                            file_path = os.path.join(base_path, audio_file)
                            data['filename'].append(file_path)
                            data['category'].append(category)
                            data['speaker_id'].append(speaker_id)
        
        # Crea il DataFrame
        self.df = pd.DataFrame(data)
        
        if len(self.df) == 0:
            raise RuntimeError(f"No audio files found in {base_path}. "
                             f"Please check if the dataset has been downloaded correctly.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (file_path, target, one_hot_target) where target is the vocal sound category
                   and one_hot_target is the one-hot encoded class vector.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)
    
    def validate_audio_files(self):
        """Rimuove file audio corrotti o non validi dal dataset"""
        import torchaudio
        
        valid_indices = []
        corrupted_files = []
        
        print("Validating audio files...")
        for idx, file_path in enumerate(tqdm(self.audio_paths, desc="Validating")):
            try:
                # Prova a caricare effettivamente il file
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Verifica che abbia dati validi
                if waveform.numel() > 0 and sample_rate > 0:
                    valid_indices.append(idx)
                else:
                    corrupted_files.append((file_path, "Empty waveform"))
            except RuntimeError as e:
                # Errore specifico di decodifica
                corrupted_files.append((file_path, str(e)))
            except Exception as e:
                # Altri errori
                corrupted_files.append((file_path, str(e)))
        
        # Filtra solo i file validi
        self.audio_paths = [self.audio_paths[i] for i in valid_indices]
        self.targets = [self.targets[i] for i in valid_indices]
        
        # Aggiorna anche il DataFrame
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        if corrupted_files:
            print(f"\n⚠️  Found {len(corrupted_files)} corrupted/invalid files:")
            # Raggruppa per tipo di errore
            error_types = {}
            for f, err in corrupted_files:
                error_key = err.split(':')[0] if ':' in err else err
                if error_key not in error_types:
                    error_types[error_key] = []
                error_types[error_key].append(os.path.basename(f))
            
            for error_type, files in error_types.items():
                print(f"\n   Error type: {error_type}")
                print(f"   Affected files: {len(files)}")
                for f in files[:3]:  # Mostra solo i primi 3 per tipo
                    print(f"      - {f}")
                if len(files) > 3:
                    print(f"      ... and {len(files) - 3} more")
        
        print(f"\n✓ Valid files: {len(self.audio_paths)}/{len(self.audio_paths) + len(corrupted_files)}")
        
        if len(valid_indices) == 0:
            raise RuntimeError("No valid audio files found! Please check your dataset.")

    def download(self):
        """Scarica ed estrae il dataset VocalSound"""
        import requests
        from zipfile import ZipFile
        
        file = Path(self.root) / self.filename
        
        # Controlla se il dataset è già stato scaricato ed estratto
        extracted_path = Path(self.root) / self.base_folder
        if extracted_path.is_dir() and any(extracted_path.iterdir()):
            print(f"Dataset già presente in {extracted_path}")
            return
            
        if file.is_file():
            print(f"File {self.filename} già scaricato")
        else:
            print(f"Downloading {self.filename}...")
            print(f"Note: Please manually download the VocalSound dataset from the official repository")
            print(f"and place the audio files in {extracted_path}")
            
            # Se l'URL è disponibile, scarica automaticamente
            try:
                r = requests.get(self.url, stream=True, timeout=10)
                r.raise_for_status()
                
                # Scarica in un file temporaneo per evitare download parziali
                tmp = file.with_suffix('.tmp')
                tmp.parent.mkdir(parents=True, exist_ok=True)
                
                with open(tmp, 'wb') as f:
                    pbar = tqdm(unit=" MB", bar_format=f'{file.name}: {{rate_noinv_fmt}}')
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            pbar.update(len(chunk) / 1024 / 1024)
                            f.write(chunk)
                
                # Sposta il file temporaneo nella posizione corretta
                tmp.rename(file)
                print(f"Download completato: {file}")
            except Exception as e:
                print(f"Automatic download failed: {e}")
                print(f"Please manually download and extract the dataset.")
                extracted_path.mkdir(parents=True, exist_ok=True)
                return
        
        # Estrae il file zip
        if file.is_file():
            print(f"Extracting {self.filename}...")
            with ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(path=self.root)
            print(f"Estrazione completata in {self.root}")
