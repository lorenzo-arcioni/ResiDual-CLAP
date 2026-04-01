from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
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


class IRMAS(AudioDataset):
    """
    IRMAS: a dataset for instrument recognition in musical audio signals.
    https://zenodo.org/records/1290750

    Struttura attesa dopo l'estrazione:
        <root>/IRMAS-TrainingData/
            cel/  cla/  flu/  gac/  gel/  org/  pia/  sax/  tru/  vio/  voi/

    Ogni file .wav ha un nome tipo: 040__[cla][nod][cla]0233__3.wav
      - [dru] / [nod]  → presenza o assenza di batteria
      - la cartella padre determina lo strumento principale (label)
    """

    base_folder = "IRMAS-TrainingData"
    url = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
    filename = "IRMAS-TrainingData.zip"

    INSTRUMENT_NAMES = {
        "cel": "cello",
        "cla": "clarinet",
        "flu": "flute",
        "gac": "guitar acoustic",
        "gel": "guitar electric",
        "org": "organ",
        "pia": "piano",
        "sax": "saxophone",
        "tru": "trumpet",
        "vio": "violin",
        "voi": "voice",
    }

    def __init__(
        self,
        root: str,
        reading_transformations: nn.Module = None,
        download: bool = True,
        use_drums_label: bool = False,
        validate: bool = True,
    ):
        super().__init__(root, download=download)

        self.use_drums_label = use_drums_label
        self.pre_transformations = reading_transformations

        self._build_class_index()
        self._load_files()

        if validate:
            self.validate_audio_files()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_class_index(self):
        self.classes = sorted(self.INSTRUMENT_NAMES.values())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self._abbr_to_name = {k: v for k, v in self.INSTRUMENT_NAMES.items()}

    def _load_files(self):
        base_path = Path(self.root) / self.base_folder

        self.audio_paths: list[str] = []
        self.targets: list[str] = []
        self.has_drums: list[bool] = []

        print("Loading IRMAS audio files...")
        for abbr, full_name in tqdm(self.INSTRUMENT_NAMES.items(), dynamic_ncols=True):
            instr_dir = base_path / abbr
            if not instr_dir.is_dir():
                print(f"  [WARNING] Cartella non trovata: {instr_dir}")
                continue

            wav_files = sorted(instr_dir.glob("*.wav"))
            for wav_path in wav_files:
                self.audio_paths.append(str(wav_path))
                self.targets.append(full_name)
                self.has_drums.append("[dru]" in wav_path.name)

        print(f"  Totale campioni caricati: {len(self.audio_paths)}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int):
        file_path = self.audio_paths[index]
        target = self.targets[index]

        idx = torch.tensor(self.class_to_idx[target])
        one_hot = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1, -1)

        if self.use_drums_label:
            return file_path, target, one_hot, self.has_drums[index]
        return file_path, target, one_hot

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
            self.has_drums   = [self.has_drums[i]   for i in valid_indices]
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
        self.targets     = [self.targets[i]   for i in valid_indices]
        self.has_drums   = [self.has_drums[i] for i in valid_indices]

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

    # ------------------------------------------------------------------
    # Utility: split per strumento
    # ------------------------------------------------------------------

    def split_by_instrument(self) -> dict[str, list[int]]:
        from collections import defaultdict
        mapping = defaultdict(list)
        for i, label in enumerate(self.targets):
            mapping[label].append(i)
        return dict(mapping)

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        import random
        rng = random.Random(seed)

        train_idx, val_idx, test_idx = [], [], []

        for indices in self.split_by_instrument().values():
            shuffled = indices[:]
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train: n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val:])

        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self):
        import requests
        from zipfile import ZipFile

        root = Path(self.root)
        extracted_dir = root / self.base_folder
        zip_file = root / self.filename

        if extracted_dir.is_dir():
            print(f"Dataset già presente in {extracted_dir}, skip download.")
            return

        if not zip_file.is_file():
            print(f"Scaricando {self.url} ...")
            r = requests.get(self.url, stream=True)
            total_size = int(r.headers.get("content-length", 0))

            tmp = zip_file.with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=zip_file.name,
                    dynamic_ncols=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            tmp.rename(zip_file)

        print(f"Estraendo {zip_file} in {root} ...")
        with ZipFile(zip_file, "r") as z:
            members = z.namelist()
            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                z.extract(member, path=root)
        print("Download e estrazione completati.")
