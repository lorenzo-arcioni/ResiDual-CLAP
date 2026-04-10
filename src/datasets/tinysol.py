from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import torch.nn as nn
import torch
import json
from datetime import datetime


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

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True, validate: bool = True):
        super().__init__(root, download=download)
        self._load_meta()
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations

        print("Loading audio files")
        self.df['category'] = self.df['category'].str.replace('_', ' ')

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), dynamic_ncols=True):
            file_path = row[self.file_col]
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

        if validate:
            self.validate_audio_files()

    def _load_meta(self):
        """Crea il DataFrame esplorando la struttura delle cartelle"""
        base_path = os.path.join(self.root, self.base_folder)

        data = {
            'filename': [],
            'category': [],
            'family': [],
            'target': []
        }

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

        self.classes = [x.replace('_', ' ') for x in sorted(all_instruments)]
        self.class_to_idx = {instrument: i for i, instrument in enumerate(self.classes)}

        for family in sorted(os.listdir(base_path)):
            family_path = os.path.join(base_path, family)
            if not os.path.isdir(family_path):
                continue
            for instrument in sorted(os.listdir(family_path)):
                instrument_path = os.path.join(family_path, instrument)
                if not os.path.isdir(instrument_path):
                    continue

                instrument_name = instrument.replace('_', ' ')
                idx = self.class_to_idx[instrument_name]

                one_hot = torch.zeros(len(self.classes))
                one_hot[idx] = 1

                ordinario_path = os.path.join(instrument_path, 'ordinario')
                if os.path.isdir(ordinario_path):
                    for audio_file in sorted(os.listdir(ordinario_path)):
                        if any(audio_file.endswith(ext) for ext in self.audio_extensions):
                            file_path = os.path.join(ordinario_path, audio_file)
                            data['filename'].append(file_path)
                            data['category'].append(instrument)
                            data['family'].append(family)
                            data['target'].append(one_hot)

        self.df = pd.DataFrame(data)

    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    @property
    def _cache_path(self) -> Path:
        return Path(self.root) / self.base_folder / '.validation_cache.json'

    def _load_cache(self) -> dict | None:
        """Carica la cache di validazione se esiste. Ritorna None se non presente o corrotta."""
        if not self._cache_path.is_file():
            return None
        try:
            with open(self._cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, valid_paths: list[str], corrupted: list[tuple[str, str]]):
        """Salva l'esito della validazione su disco."""
        cache = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.__class__.__name__,
            'valid_count': len(valid_paths),
            'corrupted_count': len(corrupted),
            'valid_paths': valid_paths,
            'corrupted': [{'path': p, 'error': e} for p, e in corrupted],
        }
        with open(self._cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"   Cache salvata in {self._cache_path}")

    def validate_audio_files(self):
        """Rimuove file audio corrotti o non validi dal dataset.
        Se esiste una cache di validazione precedente, la usa direttamente."""
        import torchaudio

        cache = self._load_cache()
        if cache is not None:
            print(f"✓ Cache di validazione trovata ({cache['timestamp']}): "
                  f"{cache['valid_count']} validi, {cache['corrupted_count']} corrotti. "
                  f"Skip validazione.")
            valid_set = set(cache['valid_paths'])
            valid_indices = [i for i, p in enumerate(self.audio_paths) if p in valid_set]
            self.audio_paths = [self.audio_paths[i] for i in valid_indices]
            self.targets     = [self.targets[i]     for i in valid_indices]
            self.df          = self.df.iloc[valid_indices].reset_index(drop=True)
            return

        valid_indices = []
        corrupted_files = []

        print("Validating audio files...")
        for idx, file_path in enumerate(tqdm(self.audio_paths, desc="Validating", dynamic_ncols=True)):
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                if waveform.numel() > 0 and sample_rate > 0:
                    valid_indices.append(idx)
                else:
                    corrupted_files.append((file_path, "Empty waveform"))
            except Exception as e:
                corrupted_files.append((file_path, str(e)))

        valid_paths = [self.audio_paths[i] for i in valid_indices]
        self._save_cache(valid_paths, corrupted_files)

        self.audio_paths = valid_paths
        self.targets     = [self.targets[i] for i in valid_indices]
        self.df          = self.df.iloc[valid_indices].reset_index(drop=True)

        if corrupted_files:
            print(f"\n⚠️  Trovati {len(corrupted_files)} file corrotti o non validi:")
            error_types = {}
            for f, err in corrupted_files:
                error_key = err.split(':')[0] if ':' in err else err
                error_types.setdefault(error_key, []).append(os.path.basename(f))
            for error_type, files in error_types.items():
                print(f"\n   Errore: {error_type}")
                print(f"   File coinvolti: {len(files)}")
                for f in files[:3]:
                    print(f"      - {f}")
                if len(files) > 3:
                    print(f"      ... e altri {len(files) - 3}")

        print(f"\n✓ File validi: {len(self.audio_paths)}/{len(self.audio_paths) + len(corrupted_files)}")

        if len(valid_indices) == 0:
            raise RuntimeError("Nessun file audio valido trovato. Controlla il dataset.")

    def download(self):
        """Scarica ed estrae il dataset TinySOL"""
        import requests
        import tarfile

        file = Path(self.root) / self.filename
        extracted_path = Path(self.root) / self.base_folder

        if extracted_path.is_dir():
            print(f"Dataset già presente in {extracted_path}")
            return

        if file.is_file():
            print(f"File {self.filename} già scaricato")
        else:
            print(f"Downloading {self.filename}...")
            r = requests.get(self.url, stream=True)
            total_size = int(r.headers.get('content-length', 0))

            tmp = file.with_suffix('.tmp')
            tmp.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file.name,
                    dynamic_ncols=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            tmp.rename(file)
            print(f"Download completato: {file}")

        print(f"Extracting {self.filename}...")
        extracted_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(file, 'r:gz') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                member.name = member.name.lstrip('./')
                if member.name:
                    tar.extract(member, path=extracted_path)

        print(f"Estrazione completata in {extracted_path}")
