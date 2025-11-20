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


class TinySOL(AudioDataset):
    base_folder = 'TinySOL'
    url = "https://zenodo.org/record/3685367/files/TinySOL.tar.gz"
    filename = "TinySOL.tar.gz"
    audio_extensions = ['.wav', '.WAV']
    label_col = 'category'
    file_col = 'filename'
    
    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        
        print("Loading audio files")
        
        # Processa il DataFrame come in ESC50
        self.df['category'] = self.df['category'].str.replace('_', ' ')
        
        for _, row in tqdm(self.df.iterrows()):
            file_path = row[self.file_col]
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)
    
    def _load_meta(self):
        """Crea il DataFrame esplorando la struttura delle cartelle"""
        base_path = os.path.join(self.root, self.base_folder)
        
        # Liste per creare il DataFrame
        data = {
            'filename': [],
            'category': [],
            'family': [],
            'target': []
        }
        
        # Prima passata: raccogli tutti gli strumenti per creare il mapping
        all_instruments = set()
        for family in sorted(os.listdir(base_path)):
            family_path = os.path.join(base_path, family)
            if not os.path.isdir(family_path):
                continue
                
            for instrument in sorted(os.listdir(family_path)):
                instrument_path = os.path.join(family_path, instrument)
                if not os.path.isdir(instrument_path):
                    continue
                all_instruments.add(instrument)
        
        # Crea il mapping classe-indice
        self.classes = [x.replace('_', ' ') for x in sorted(all_instruments)]
        self.class_to_idx = {}
        for i, instrument in enumerate(self.classes):
            self.class_to_idx[instrument] = i
        
        # Seconda passata: popola il DataFrame con i target
        for family in sorted(os.listdir(base_path)):
            family_path = os.path.join(base_path, family)
            if not os.path.isdir(family_path):
                continue
                
            for instrument in sorted(os.listdir(family_path)):
                instrument_path = os.path.join(family_path, instrument)
                if not os.path.isdir(instrument_path):
                    continue
                
                # Normalizza il nome dello strumento
                instrument_name = instrument.replace('_', ' ')
                idx = self.class_to_idx[instrument_name]
                
                # Crea il one-hot encoding
                one_hot = torch.zeros(len(self.classes))
                one_hot[idx] = 1
                
                # Cerca file audio nella cartella ordinario
                ordinario_path = os.path.join(instrument_path, 'ordinario')
                if os.path.isdir(ordinario_path):
                    for audio_file in sorted(os.listdir(ordinario_path)):
                        if any(audio_file.endswith(ext) for ext in self.audio_extensions):
                            file_path = os.path.join(ordinario_path, audio_file)
                            data['filename'].append(file_path)
                            data['category'].append(instrument)
                            data['family'].append(family)
                            data['target'].append(one_hot)
        
        # Crea il DataFrame
        self.df = pd.DataFrame(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (file_path, target, one_hot_target) where target is the instrument name
                   and one_hot_target is the one-hot encoded class vector.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        """Scarica ed estrae il dataset TinySOL"""
        import requests
        import tarfile
        
        file = Path(self.root) / self.filename
        
        # Controlla se il dataset è già stato scaricato ed estratto
        extracted_path = Path(self.root) / self.base_folder
        if extracted_path.is_dir():
            print(f"Dataset già presente in {extracted_path}")
            return
            
        if file.is_file():
            print(f"File {self.filename} già scaricato")
        else:
            print(f"Downloading {self.filename}...")
            r = requests.get(self.url, stream=True)
            
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
        
        # Estrae il file tar.gz
        print(f"Extracting {self.filename}...")
        with tarfile.open(file, 'r:gz') as tar:
            tar.extractall(path=self.root)
        print(f"Estrazione completata in {self.root}")