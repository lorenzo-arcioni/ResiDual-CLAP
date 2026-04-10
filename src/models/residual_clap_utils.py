"""
residual_clap_utils.py
======================
Utility condivise per i notebook di analisi ResiDual CLAP.

Espone:
  - build_clap(type, residual_config, version, use_cuda) -> CLAPWrapper
  - build_pca_loader(wrapper, dataset, max_samples, batch_size) -> DataLoader
  - fit_pca(clap_residual, pca_loader, max_samples) -> variance_info
  - evaluate_model(wrapper, dataset, text_labels, indices, batch_size) -> (preds, labels)
  - accuracy(preds, labels) -> float
  - KFoldEvaluator
  - BayesianHPTuner
  - DATASET_REGISTRY
"""

import contextlib
import io
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
# CONTEXT MANAGER: SOPPRESSIONE OUTPUT
# ============================================================================

@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    Context manager che sopprime completamente stdout e stderr,
    incluso l'output nativo C/C++ dei modelli (reindirizza i file descriptor).
    """
    # Flush prima di redirigere
    sys.stdout.flush()
    sys.stderr.flush()

    # Salva i file descriptor originali
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Apri /dev/null
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # Reindirizza fd 1 e 2 a /dev/null
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

        # Reindirizza anche gli stream Python
        old_py_stdout = sys.stdout
        old_py_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        yield

    finally:
        # Ripristina fd originali
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        # Ripristina stream Python
        sys.stdout = old_py_stdout
        sys.stderr = old_py_stderr


@contextlib.contextmanager
def maybe_suppress(suppress: bool):
    """Applica suppress_stdout_stderr solo se suppress=True."""
    if suppress:
        with suppress_stdout_stderr():
            yield
    else:
        yield


# ============================================================================
# DATASET REGISTRY
# ============================================================================

def get_dataset_registry():
    """
    Ritorna il registry dei dataset disponibili.
    Import lazy per evitare errori se alcuni dataset non sono installati.

    Returns:
        dict: {nome: classe_dataset}
    """
    registry = {}

    try:
        from datasets.esc50 import ESC50
        registry['ESC50'] = ESC50
    except ImportError:
        pass

    try:
        from datasets.tinysol import TinySOL
        registry['TinySOL'] = TinySOL
    except ImportError:
        pass

    try:
        from datasets.vocalsound import VocalSound
        registry['VocalSound'] = VocalSound
    except ImportError:
        pass

    try:
        from datasets.irmas import IRMAS
        registry['IRMAS'] = IRMAS
    except ImportError:
        pass

    return registry


# ============================================================================
# COSTRUZIONE MODELLO
# ============================================================================

def build_clap(
    model_type: str = 'classic',
    residual_config: Optional[Dict] = None,
    version: str = '2023',
    use_cuda: bool = True
):
    """
    Costruisce e ritorna un CLAPWrapper.

    Args:
        model_type: 'classic' o 'residual'
        residual_config: Configurazione ResiDual (richiesta solo se type='residual')
        version: Versione modello CLAP ('2022' o '2023')
        use_cuda: Se True, usa GPU se disponibile

    Returns:
        CLAPWrapper inizializzato
    """
    from CLAPWrapper import CLAPWrapper

    kwargs = {
        'version': version,
        'use_cuda': use_cuda and torch.cuda.is_available(),
        'type': model_type,
    }

    if model_type == 'residual':
        if residual_config is None:
            raise ValueError("residual_config è obbligatorio per model_type='residual'")
        kwargs['residual_config'] = residual_config

    wrapper = CLAPWrapper(**kwargs)
    wrapper.clap.eval()

    return wrapper


# ============================================================================
# PCA FITTING
# ============================================================================

class _SimpleAudioDataset(torch.utils.data.Dataset):
    """Dataset interno per raccolta campioni PCA con bilanciamento per classe."""

    def __init__(self, wrapper, dataset, max_samples: int = 1000, random_state: int = 42):
        self.wrapper = wrapper
        self.audio_paths = []

        rng = np.random.RandomState(random_state)

        n_classes = len(dataset.classes)
        samples_per_class = max(1, max_samples // n_classes)

        class_indices: Dict[str, List[int]] = {cls: [] for cls in dataset.classes}
        for i in range(len(dataset)):
            _, target, _ = dataset[i]
            class_indices[target].append(i)

        for cls, indices in class_indices.items():
            indices_arr = np.array(indices)
            chosen = rng.choice(
                indices_arr,
                size=min(samples_per_class, len(indices_arr)),
                replace=False
            )
            for i in chosen:
                audio_path, _, _ = dataset[int(i)]
                self.audio_paths.append(audio_path)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_tensor = self.wrapper.load_audio_into_tensor(
            self.audio_paths[idx],
            self.wrapper.args.duration,
            resample=True
        )
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        return audio_tensor


def build_pca_loader(
    wrapper,
    dataset,
    max_samples: int = 500,
    batch_size: int = 4,
    num_workers: int = 0,
    random_state: int = 42,
) -> DataLoader:
    """
    Costruisce un DataLoader bilanciato per il fitting PCA.

    Args:
        wrapper: CLAPWrapper (residual)
        dataset: Dataset audio
        max_samples: Numero massimo di campioni totali
        batch_size: Dimensione batch
        num_workers: Worker DataLoader
        random_state: Seed per riproducibilità

    Returns:
        DataLoader pronto per fit_spectral_layers
    """
    pca_ds = _SimpleAudioDataset(
        wrapper, dataset, max_samples=max_samples, random_state=random_state
    )
    return DataLoader(
        pca_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )


def fit_pca(
    clap_residual,
    pca_loader: DataLoader,
    max_samples: int = 500
) -> Dict:
    """
    Esegue il fitting degli spectral layers.

    Args:
        clap_residual: CLAPWrapper residual
        pca_loader: DataLoader con campioni audio
        max_samples: Massimo campioni da usare

    Returns:
        variance_info: Dict con variance ratios per layer/block/head
    """
    htsat = clap_residual.clap.audio_encoder.base.htsat
    variance_info = htsat.fit_spectral_layers(pca_loader, max_samples=max_samples)
    return variance_info


# ============================================================================
# VALUTAZIONE
# ============================================================================

def get_text_embeddings(wrapper, text_labels: List[str]) -> torch.Tensor:
    """Calcola gli embedding testuali una volta sola."""
    return wrapper.get_text_embeddings(text_labels)


def evaluate_model(
    wrapper,
    dataset,
    text_labels: List[str],
    indices: Optional[List[int]] = None,
    batch_size: int = 1,
    show_progress: bool = True,
    desc: str = "Valutazione"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valuta un wrapper CLAP su un sottoinsieme del dataset.

    Args:
        wrapper: CLAPWrapper
        dataset: Dataset audio con __getitem__ -> (audio_path, target, one_hot)
        text_labels: Lista di label testuali per la classificazione
        indices: Indici da valutare (None = tutto il dataset)
        batch_size: Batch size per le query audio (default 1 per compatibilità)
        show_progress: Mostra tqdm
        desc: Descrizione barra progresso

    Returns:
        (y_preds, y_labels): Array numpy (N, n_classes) ciascuno
    """
    if indices is None:
        indices = list(range(len(dataset)))

    text_embeddings = get_text_embeddings(wrapper, text_labels)

    y_preds, y_labels = [], []

    it = tqdm(indices, desc=desc) if show_progress else indices

    for idx in it:
        audio_path, target, one_hot_target = dataset[idx]

        audio_embedding = wrapper.get_audio_embeddings([audio_path], resample=True)
        similarity = wrapper.compute_similarity(audio_embedding, text_embeddings)

        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())

    return np.concatenate(y_preds, axis=0), np.concatenate(y_labels, axis=0)


def accuracy(y_preds: np.ndarray, y_labels: np.ndarray) -> float:
    """
    Calcola accuracy da array (N, n_classes) di probabilità e one-hot.

    Returns:
        Accuracy float in [0, 1]
    """
    return accuracy_score(
        np.argmax(y_labels, axis=1),
        np.argmax(y_preds, axis=1)
    )


# ============================================================================
# STRATIFIED DATASET SUBSAMPLING
# ============================================================================

def stratified_subsample(
    label_indices: List[int],
    max_samples: int,
    random_state: int = 42,
) -> List[int]:
    """
    Esegue un campionamento stratificato riproducibile su un dataset.

    Dato un array di indici posizionali [0..N-1] con le rispettive label,
    restituisce un sottoinsieme di al più `max_samples` indici, mantenendo
    le proporzioni di classe il più possibile.

    Args:
        label_indices: Lista di label intere (lunghezza N = size del dataset).
        max_samples: Dimensione massima desiderata del sottoinsieme.
        random_state: Seed per riproducibilità.

    Returns:
        Lista di indici posizionali selezionati (ordinati).
    """
    n = len(label_indices)
    if max_samples >= n:
        return list(range(n))

    rng = np.random.RandomState(random_state)
    labels_arr = np.array(label_indices)
    classes, counts = np.unique(labels_arr, return_counts=True)

    # Distribuzione proporzionale con almeno 1 campione per classe
    proportions = counts / counts.sum()
    per_class = np.maximum(1, np.floor(proportions * max_samples).astype(int))

    # Aggiusta per non superare max_samples
    while per_class.sum() > max_samples:
        # Riduci la classe con più campioni assegnati
        idx_max = np.argmax(per_class)
        per_class[idx_max] -= 1

    selected = []
    for cls, n_sel in zip(classes, per_class):
        cls_positions = np.where(labels_arr == cls)[0]
        chosen = rng.choice(cls_positions, size=int(n_sel), replace=False)
        selected.extend(chosen.tolist())

    selected.sort()
    return selected


# ============================================================================
# K-FOLD EVALUATOR
# ============================================================================

class KFoldEvaluator:
    """
    Valutazione K-Fold di baseline CLAP e ResiDual CLAP su un dataset.

    Parametri nuovi rispetto alla versione originale:
        max_dataset_size (int | None):
            Se specificato e il dataset supera questa dimensione, viene eseguito
            un campionamento stratificato riproducibile prima di formare i fold.
            In questo modo il dataset effettivo usato per train/test non supera
            mai `max_dataset_size` campioni totali.
        suppress_model_prints (bool):
            Se True, sopprime stdout e stderr dei modelli caricati internamente
            (incluso l'output nativo C/C++). Default: True.

    Uso:
        evaluator = KFoldEvaluator(
            dataset=dataset,
            text_labels=text_labels,
            residual_config=residual_config,
            n_splits=5,
            pca_samples=300,
            test_size=100,
            max_dataset_size=500,       # ← nuovo
            suppress_model_prints=True, # ← nuovo
        )
        results = evaluator.run()
    """

    def __init__(
        self,
        dataset,
        text_labels: List[str],
        residual_config: Dict,
        n_splits: int = 5,
        pca_samples: int = 300,
        test_size: Optional[int] = None,
        max_dataset_size: Optional[int] = None,
        suppress_model_prints: bool = True,
        clap_version: str = '2023',
        use_cuda: bool = True,
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.text_labels = text_labels
        self.residual_config = residual_config
        self.n_splits = n_splits
        self.pca_samples = pca_samples
        self.test_size = test_size
        self.max_dataset_size = max_dataset_size
        self.suppress_model_prints = suppress_model_prints
        self.clap_version = clap_version
        self.use_cuda = use_cuda
        self.random_state = random_state

        # Estrai le label intere per StratifiedKFold
        self.all_labels: List[str] = []
        for i in range(len(dataset)):
            _, target, _ = dataset[i]
            self.all_labels.append(target)

        self.class_to_idx = dataset.class_to_idx
        self.label_indices: List[int] = [self.class_to_idx[l] for l in self.all_labels]

    def _get_effective_indices(self) -> List[int]:
        """
        Restituisce gli indici effettivi da usare per i fold.
        Se max_dataset_size è impostato e il dataset è più grande,
        esegue un campionamento stratificato riproducibile.
        """
        n = len(self.dataset)
        if self.max_dataset_size is None or n <= self.max_dataset_size:
            return list(range(n))

        selected = stratified_subsample(
            label_indices=self.label_indices,
            max_samples=self.max_dataset_size,
            random_state=self.random_state,
        )
        return selected

    def run(self, verbose: bool = True) -> Dict:
        """
        Esegue la valutazione K-Fold completa.

        Returns:
            results: {
                'baseline': {
                    'fold_accs': [float, ...],
                    'mean': float,
                    'std': float,
                },
                'residual': {
                    'fold_accs': [float, ...],
                    'mean': float,
                    'std': float,
                },
                'delta_mean': float,   # residual - baseline
                'n_splits': int,
                'effective_dataset_size': int,
            }
        """
        # --- Determina gli indici effettivi (con eventuale subsampling) ---
        effective_indices = self._get_effective_indices()
        effective_labels = np.array([self.label_indices[i] for i in effective_indices])
        effective_indices_arr = np.array(effective_indices)

        if verbose:
            n_orig = len(self.dataset)
            n_eff = len(effective_indices)
            if n_eff < n_orig:
                print(f"  ⚠️  Dataset ridotto: {n_orig} → {n_eff} campioni "
                      f"(max_dataset_size={self.max_dataset_size}, "
                      f"campionamento stratificato, seed={self.random_state})")

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        baseline_accs = []
        residual_accs = []

        for fold_idx, (train_pos, test_pos) in enumerate(
            skf.split(effective_indices_arr, effective_labels)
        ):
            # Converti posizioni nel subset → indici reali nel dataset originale
            train_idx = effective_indices_arr[train_pos]
            test_idx = effective_indices_arr[test_pos]

            if verbose:
                print(f"\n{'='*60}")
                print(f"FOLD {fold_idx + 1} / {self.n_splits}")
                print(f"{'='*60}")
                print(f"  Train: {len(train_idx)} campioni | Test: {len(test_idx)} campioni")

            # Limita test_size se specificato
            if self.test_size is not None and len(test_idx) > self.test_size:
                rng = np.random.RandomState(self.random_state + fold_idx)
                test_idx = rng.choice(test_idx, size=self.test_size, replace=False)

            # --- BASELINE ---
            if verbose:
                print(f"\n  [Baseline] Caricamento modello...")

            with maybe_suppress(self.suppress_model_prints):
                clap_std = build_clap(
                    model_type='classic',
                    version=self.clap_version,
                    use_cuda=self.use_cuda
                )

            y_preds_b, y_labels = evaluate_model(
                wrapper=clap_std,
                dataset=self.dataset,
                text_labels=self.text_labels,
                indices=test_idx.tolist(),
                desc=f"  Fold {fold_idx+1} Baseline"
            )
            acc_b = accuracy(y_preds_b, y_labels)
            baseline_accs.append(acc_b)
            if verbose:
                print(f"  Baseline accuracy: {acc_b:.4f}")

            del clap_std
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- RESIDUAL ---
            if verbose:
                print(f"\n  [Residual] Caricamento modello...")

            with maybe_suppress(self.suppress_model_prints):
                clap_res = build_clap(
                    model_type='residual',
                    residual_config=self.residual_config,
                    version=self.clap_version,
                    use_cuda=self.use_cuda
                )

            # PCA fitting su train split
            if verbose:
                print(f"  [Residual] Fitting PCA su {self.pca_samples} campioni da train split...")

            train_dataset_proxy = _IndexedSubset(self.dataset, train_idx.tolist())

            with maybe_suppress(self.suppress_model_prints):
                pca_loader = build_pca_loader(
                    wrapper=clap_res,
                    dataset=train_dataset_proxy,
                    max_samples=self.pca_samples,
                    random_state=self.random_state + fold_idx * 1000,
                )
                fit_pca(clap_res, pca_loader, max_samples=self.pca_samples)

            y_preds_r, _ = evaluate_model(
                wrapper=clap_res,
                dataset=self.dataset,
                text_labels=self.text_labels,
                indices=test_idx.tolist(),
                desc=f"  Fold {fold_idx+1} Residual"
            )
            acc_r = accuracy(y_preds_r, y_labels)
            residual_accs.append(acc_r)
            if verbose:
                print(f"  Residual accuracy: {acc_r:.4f}")
                print(f"  Delta: {acc_r - acc_b:+.4f}")

            del clap_res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results = {
            'baseline': {
                'fold_accs': baseline_accs,
                'mean': float(np.mean(baseline_accs)),
                'std': float(np.std(baseline_accs)),
            },
            'residual': {
                'fold_accs': residual_accs,
                'mean': float(np.mean(residual_accs)),
                'std': float(np.std(residual_accs)),
            },
            'delta_mean': float(np.mean(residual_accs) - np.mean(baseline_accs)),
            'n_splits': self.n_splits,
            'effective_dataset_size': len(effective_indices),
        }

        if verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        print(f"\n{'='*60}")
        print("RIEPILOGO K-FOLD")
        print(f"{'='*60}")
        b = results['baseline']
        r = results['residual']
        print(f"Baseline:  {b['mean']:.4f} ± {b['std']:.4f}  (folds: {[f'{x:.3f}' for x in b['fold_accs']]})")
        print(f"Residual:  {r['mean']:.4f} ± {r['std']:.4f}  (folds: {[f'{x:.3f}' for x in r['fold_accs']]})")
        print(f"Delta:     {results['delta_mean']:+.4f}")
        print(f"Campioni effettivi usati: {results['effective_dataset_size']}")


class _IndexedSubset:
    """Proxy per rendere un subset del dataset compatibile con _SimpleAudioDataset."""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# ============================================================================
# BAYESIAN HYPERPARAMETER TUNER
# ============================================================================

class BayesianHPTuner:
    """
    Ottimizzazione bayesiana degli iperparametri ResiDual CLAP.

    Parametri ottimizzabili:
        - n_components_ratio: float in [lo, hi]
        - target_layers:      lista di layer da [0, 1, 2, 3]

    Usa scikit-optimize (skopt) per la Bayesian optimization.

    Parametri aggiuntivi:
        suppress_model_prints (bool): Sopprime l'output dei modelli interni.

    Uso:
        tuner = BayesianHPTuner(
            dataset=dataset,
            text_labels=text_labels,
            base_residual_config=base_config,
            param_space={
                'n_components_ratio': (0.05, 0.5),
                'target_layers_options': [[0],[1],[2],[3],[0,1],[1,2],[2,3],[0,1,2],[1,2,3],[0,1,2,3]],
            },
            n_calls=20,
            n_initial_points=5,
            eval_samples=150,
            pca_samples=200,
            suppress_model_prints=True,
        )
        best_params, best_score, all_results = tuner.run()
    """

    def __init__(
        self,
        dataset,
        text_labels: List[str],
        base_residual_config: Dict,
        param_space: Dict,
        n_calls: int = 20,
        n_initial_points: int = 5,
        eval_samples: int = 150,
        pca_samples: int = 200,
        suppress_model_prints: bool = True,
        clap_version: str = '2023',
        use_cuda: bool = True,
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.text_labels = text_labels
        self.base_config = base_residual_config
        self.param_space = param_space
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.eval_samples = eval_samples
        self.pca_samples = pca_samples
        self.suppress_model_prints = suppress_model_prints
        self.clap_version = clap_version
        self.use_cuda = use_cuda
        self.random_state = random_state

        self.all_results: List[Dict] = []
        self._call_count = 0

        # Baseline score (calcolato una volta)
        self._baseline_acc: Optional[float] = None

    def _get_baseline(self) -> float:
        if self._baseline_acc is not None:
            return self._baseline_acc

        print("Calcolo baseline (una tantum)...")

        with maybe_suppress(self.suppress_model_prints):
            clap_std = build_clap(
                model_type='classic',
                version=self.clap_version,
                use_cuda=self.use_cuda
            )

        indices = np.random.RandomState(self.random_state).choice(
            len(self.dataset),
            size=min(self.eval_samples, len(self.dataset)),
            replace=False
        ).tolist()

        y_preds, y_labels = evaluate_model(
            wrapper=clap_std,
            dataset=self.dataset,
            text_labels=self.text_labels,
            indices=indices,
            desc="Baseline",
        )
        self._baseline_acc = accuracy(y_preds, y_labels)
        print(f"Baseline accuracy: {self._baseline_acc:.4f}")

        del clap_std
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._baseline_acc

    def _objective(self, params_list) -> float:
        """
        Funzione obiettivo per skopt.
        params_list = [n_components_ratio, target_layers_idx]
        Ritorna il NEGATIVO dell'accuracy (skopt minimizza).
        """
        self._call_count += 1
        n_comp_ratio = float(params_list[0])
        layers_idx = int(params_list[1])
        target_layers = self.param_space['target_layers_options'][layers_idx]

        print(f"\n--- Trial {self._call_count}/{self.n_calls} ---")
        print(f"  n_components_ratio = {n_comp_ratio:.4f}")
        print(f"  target_layers      = {target_layers}")

        config = dict(self.base_config)
        config['n_components_ratio'] = n_comp_ratio
        config['target_layers'] = target_layers

        try:
            with maybe_suppress(self.suppress_model_prints):
                clap_res = build_clap(
                    model_type='residual',
                    residual_config=config,
                    version=self.clap_version,
                    use_cuda=self.use_cuda
                )

            with maybe_suppress(self.suppress_model_prints):
                pca_loader = build_pca_loader(
                    wrapper=clap_res,
                    dataset=self.dataset,
                    max_samples=self.pca_samples,
                    random_state=self.random_state + self._call_count,
                )
                fit_pca(clap_res, pca_loader, max_samples=self.pca_samples)

            rng = np.random.RandomState(self.random_state + self._call_count)
            indices = rng.choice(
                len(self.dataset),
                size=min(self.eval_samples, len(self.dataset)),
                replace=False
            ).tolist()

            y_preds, y_labels = evaluate_model(
                wrapper=clap_res,
                dataset=self.dataset,
                text_labels=self.text_labels,
                indices=indices,
                desc=f"  Trial {self._call_count}",
            )
            acc = accuracy(y_preds, y_labels)

        except Exception as e:
            print(f"  ERRORE: {e}")
            acc = 0.0

        finally:
            try:
                del clap_res
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        delta = acc - (self._baseline_acc or 0.0)
        print(f"  Accuracy: {acc:.4f}  (delta vs baseline: {delta:+.4f})")

        result = {
            'trial': self._call_count,
            'n_components_ratio': n_comp_ratio,
            'target_layers': target_layers,
            'accuracy': acc,
            'delta_vs_baseline': delta,
            'params_list': params_list,
        }
        self.all_results.append(result)

        return -acc  # minimize negative accuracy

    def run(self, verbose: bool = True) -> Tuple[Dict, float, List[Dict]]:
        """
        Esegue la Bayesian optimization.

        Returns:
            best_params: {'n_components_ratio': float, 'target_layers': list}
            best_score:  float (accuracy)
            all_results: lista di tutti i trial
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            raise ImportError(
                "scikit-optimize non installato. "
                "Esegui: pip install scikit-optimize"
            )

        # Calcola baseline prima
        self._get_baseline()

        n_layer_options = len(self.param_space['target_layers_options'])
        n_comp_lo, n_comp_hi = self.param_space.get('n_components_ratio', (0.05, 0.5))

        search_space = [
            Real(n_comp_lo, n_comp_hi, name='n_components_ratio', prior='uniform'),
            Integer(0, n_layer_options - 1, name='target_layers_idx'),
        ]

        if verbose:
            print(f"\n{'='*60}")
            print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
            print(f"{'='*60}")
            print(f"  n_calls:          {self.n_calls}")
            print(f"  n_initial_points: {self.n_initial_points}")
            print(f"  eval_samples:     {self.eval_samples}")
            print(f"  pca_samples:      {self.pca_samples}")
            print(f"  n_comp_ratio:     [{n_comp_lo}, {n_comp_hi}]")
            print(f"  target_layers options: {self.param_space['target_layers_options']}")
            print(f"{'='*60}\n")

        result = gp_minimize(
            func=self._objective,
            dimensions=search_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=False,
        )

        best_n_comp = float(result.x[0])
        best_layers_idx = int(result.x[1])
        best_layers = self.param_space['target_layers_options'][best_layers_idx]
        best_acc = -result.fun

        best_params = {
            'n_components_ratio': best_n_comp,
            'target_layers': best_layers,
        }

        if verbose:
            print(f"\n{'='*60}")
            print("RISULTATI OTTIMIZZAZIONE")
            print(f"{'='*60}")
            print(f"  Best n_components_ratio: {best_n_comp:.4f}")
            print(f"  Best target_layers:      {best_layers}")
            print(f"  Best accuracy:           {best_acc:.4f}")
            print(f"  Baseline accuracy:       {self._baseline_acc:.4f}")
            print(f"  Delta:                   {best_acc - self._baseline_acc:+.4f}")

        return best_params, best_acc, self.all_results


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_kfold_results(results_by_dataset: Dict[str, Dict], figsize=(12, 5)):
    """
    Plotta i risultati K-Fold per più dataset.

    Args:
        results_by_dataset: {dataset_name: kfold_results}
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dataset_names = list(results_by_dataset.keys())
    n_ds = len(dataset_names)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Mean accuracy con std ---
    ax = axes[0]
    x = np.arange(n_ds)
    width = 0.35

    baseline_means = [results_by_dataset[d]['baseline']['mean'] for d in dataset_names]
    baseline_stds  = [results_by_dataset[d]['baseline']['std']  for d in dataset_names]
    residual_means = [results_by_dataset[d]['residual']['mean'] for d in dataset_names]
    residual_stds  = [results_by_dataset[d]['residual']['std']  for d in dataset_names]

    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline CLAP', color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, residual_means, width, yerr=residual_stds,
                   label='ResiDual CLAP', color='coral', alpha=0.8, capsize=5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy media K-Fold ± std')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # --- Plot 2: Delta per fold ---
    ax2 = axes[1]

    for i, ds_name in enumerate(dataset_names):
        b_folds = results_by_dataset[ds_name]['baseline']['fold_accs']
        r_folds = results_by_dataset[ds_name]['residual']['fold_accs']
        deltas = [r - b for r, b in zip(r_folds, b_folds)]
        fold_xs = np.arange(len(deltas))
        ax2.plot(fold_xs, deltas, marker='o', label=ds_name, alpha=0.8)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Delta accuracy (Residual - Baseline)')
    ax2.set_title('Differenza per fold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_hp_results(all_results: List[Dict], baseline_acc: float, figsize=(14, 5)):
    """
    Plotta i risultati dell'ottimizzazione iperparametri.

    Args:
        all_results: Lista di trial da BayesianHPTuner
        baseline_acc: Accuracy baseline
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    trials     = [r['trial']              for r in all_results]
    accs       = [r['accuracy']           for r in all_results]
    n_comps    = [r['n_components_ratio'] for r in all_results]
    layers_str = [str(r['target_layers']) for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Accuracy per trial
    ax = axes[0]
    ax.plot(trials, accs, 'o-', color='coral', alpha=0.8, label='Residual')
    ax.axhline(baseline_acc, color='steelblue', linestyle='--', label=f'Baseline ({baseline_acc:.3f})')
    best_so_far = [max(accs[:i+1]) for i in range(len(accs))]
    ax.plot(trials, best_so_far, 'g--', alpha=0.6, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per trial')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. n_components_ratio vs accuracy
    ax2 = axes[1]
    scatter = ax2.scatter(n_comps, accs, c=accs, cmap='RdYlGn', s=60, alpha=0.8, zorder=3)
    ax2.axhline(baseline_acc, color='steelblue', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='Accuracy')
    ax2.set_xlabel('n_components_ratio')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('n_components_ratio vs Accuracy')
    ax2.grid(alpha=0.3)

    # 3. target_layers vs accuracy (boxplot per combinazione)
    ax3 = axes[2]
    layers_acc: Dict = defaultdict(list)
    for r in all_results:
        layers_acc[str(r['target_layers'])].append(r['accuracy'])

    labels_sorted = sorted(layers_acc.keys())
    data_sorted = [layers_acc[k] for k in labels_sorted]

    bp = ax3.boxplot(data_sorted, labels=labels_sorted, patch_artist=True, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightyellow')
    ax3.axhline(baseline_acc, color='steelblue', linestyle='--', alpha=0.6, label='Baseline')
    ax3.set_xlabel('target_layers')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('target_layers vs Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def print_hp_table(all_results: List[Dict], top_n: int = 10):
    """Stampa una tabella ordinata dei migliori trial."""
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)

    print(f"\n{'='*65}")
    print(f"TOP {top_n} CONFIGURAZIONI")
    print(f"{'='*65}")
    print(f"{'#':<4} {'n_comp_ratio':<14} {'target_layers':<20} {'accuracy':<10} {'delta'}")
    print(f"{'-'*65}")

    for i, r in enumerate(sorted_results[:top_n], 1):
        print(f"{i:<4} {r['n_components_ratio']:<14.4f} "
              f"{str(r['target_layers']):<20} "
              f"{r['accuracy']:<10.4f} "
              f"{r['delta_vs_baseline']:+.4f}")