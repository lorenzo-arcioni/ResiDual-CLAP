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


class VocalSound(AudioDataset):
    base_folder = 'VocalSound'
    url = "https://huggingface.co/datasets/lorenzo-arcioni/vocalsound-test/resolve/main/VocalSound.zip"
    filename = "vocalsound.zip"
    audio_extensions = ['.wav', '.WAV']
    label_col = 'category'
    file_col = 'filename'

    vocal_categories = ['cough', 'laughter', 'sigh', 'sneeze', 'sniff', 'throatclearing']

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True, validate: bool = True):
        super().__init__(root, download=download)
        self._load_meta()
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations

        print("Loading audio files")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), dynamic_ncols=True):
            self.targets.append(row[self.label_col])
            self.audio_paths.append(row[self.file_col])

        if validate:
            self.validate_audio_files()

    def _load_meta(self):
        """Crea il DataFrame esplorando la cartella data_44k del dataset"""
        base_path = Path(self.root) / self.base_folder / 'data_44k'

        if not base_path.is_dir():
            raise RuntimeError(
                f"Cartella data_44k non trovata in {base_path}. "
                f"Verifica che il dataset sia stato estratto correttamente."
            )

        self.classes = sorted(self.vocal_categories)
        self.class_to_idx = {cat: i for i, cat in enumerate(self.classes)}

        data = {'filename': [], 'category': [], 'speaker_id': []}

        for audio_file in sorted(base_path.iterdir()):
            if audio_file.suffix.lower() not in [ext.lower() for ext in self.audio_extensions]:
                continue

            # Formato: speakerID_index_category.wav (es. f0003_0_cough.wav)
            stem_parts = audio_file.stem.rsplit('_', 1)
            if len(stem_parts) != 2:
                continue

            category = stem_parts[1]
            if category not in self.vocal_categories:
                continue

            speaker_id = audio_file.stem.split('_')[0]

            data['filename'].append(str(audio_file))
            data['category'].append(category)
            data['speaker_id'].append(speaker_id)

        self.df = pd.DataFrame(data)

        if len(self.df) == 0:
            raise RuntimeError(
                f"Nessun file audio trovato in {base_path}. "
                f"Verifica che il dataset sia stato scaricato correttamente."
            )

    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    # ------------------------------------------------------------------
    # Validation + cache
    # ------------------------------------------------------------------

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
        """Scarica ed estrae il dataset VocalSound"""
        import requests
        from zipfile import ZipFile

        file = Path(self.root) / self.filename
        extracted_path = Path(self.root) / self.base_folder

        if extracted_path.is_dir() and any(extracted_path.iterdir()):
            print(f"Dataset già presente in {extracted_path}")
            return

        if file.is_file():
            print(f"File {self.filename} già scaricato")
        else:
            print(f"Downloading {self.filename}...")
            try:
                r = requests.get(self.url, stream=True, timeout=30)
                r.raise_for_status()

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
            except Exception as e:
                print(f"Download fallito: {e}")
                print(f"Scarica manualmente il dataset e posizionalo in {extracted_path}")
                extracted_path.mkdir(parents=True, exist_ok=True)
                return

        # Estrae lo zip in data/VocalSound/
        print(f"Extracting {self.filename}...")
        extracted_path.mkdir(parents=True, exist_ok=True)

        with ZipFile(file, 'r') as zip_ref:
            members = zip_ref.namelist()
            prefix = members[0].split('/')[0] + '/' if '/' in members[0] else ''

            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                relative = member[len(prefix):] if member.startswith(prefix) else member
                if not relative:
                    continue

                dest = extracted_path / relative
                if member.endswith('/'):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zip_ref.read(member))

        print(f"Estrazione completata in {extracted_path}")
