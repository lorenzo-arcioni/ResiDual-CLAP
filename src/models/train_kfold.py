"""
train_kfold.py
==============
Utility per K-Fold cross-validation di CLAP (classico) e ResiDualCLAP.

Pipeline per ogni fold:
    1. Split dataset in K fold (stratificato, deterministico)
    2. Il fold di training viene diviso in:
        - pca_samples   → fitting PCA (solo ResiDual)
        - lambda_samples → ottimizzazione λ (solo ResiDual)
    3. Il fold di test viene usato per la valutazione finale
    4. I risultati vengono salvati in kfolds/<dataset_name>/

Uso:
    from train_kfold import run_kfold_residual, run_kfold_classic
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime


# =============================================================================
# SEED — tutto deterministico
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CAMPIONAMENTO STRATIFICATO
# =============================================================================

def stratified_sample(dataset, n_samples: int, seed: int = 42) -> list[int]:
    """
    Restituisce al massimo n_samples indici scelti con campionamento stratificato.
    Se il dataset è più piccolo di n_samples, restituisce tutti gli indici.
    Completamente riproducibile tramite seed.
    """
    targets = [dataset.targets[i] for i in range(len(dataset))]
    classes = sorted(set(targets))
    class_to_indices = {c: [] for c in classes}
    for idx, t in enumerate(targets):
        class_to_indices[t].append(idx)

    if len(dataset) <= n_samples:
        return list(range(len(dataset)))

    rng = random.Random(seed)
    samples_per_class = n_samples // len(classes)
    remainder = n_samples % len(classes)

    selected = []
    for i, cls in enumerate(classes):
        indices = class_to_indices[cls][:]
        rng.shuffle(indices)
        n = samples_per_class + (1 if i < remainder else 0)
        selected.extend(indices[:n])

    rng.shuffle(selected)
    return selected


# =============================================================================
# COLLATE — converte path audio in tensori usando CLAPWrapper
# =============================================================================

def make_collate_fn(clap_wrapper, use_cuda: bool):
    """Restituisce una funzione collate che carica l'audio da file path."""
    def collate_fn(batch):
        # batch: lista di (file_path, target_str, one_hot)
        paths   = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        # Indice numerico della classe
        labels = torch.tensor(
            [clap_wrapper.clap.audio_encoder.base.htsat.num_classes  # placeholder
             if False else
             list(batch[0][2].squeeze().shape)[0]  # fallback
             for _ in batch],
            dtype=torch.long
        )
        # Ricaviamo label numerica dalla one_hot
        labels = torch.stack([item[2].squeeze() for item in batch]).argmax(dim=-1)

        # Carichiamo audio
        audio = clap_wrapper.preprocess_audio(paths, resample=True)

        if use_cuda and torch.cuda.is_available():
            audio = audio.cuda()
        return audio, labels
    return collate_fn


def make_audio_only_collate(clap_wrapper, use_cuda: bool):
    """Collate che restituisce solo l'audio (per PCA fitting)."""
    def collate_fn(batch):
        paths = [item[0] for item in batch]
        audio = clap_wrapper.preprocess_audio(paths, resample=True).squeeze(1)
        if use_cuda and torch.cuda.is_available():
            audio = audio.cuda()
        return audio
    return collate_fn


# =============================================================================
# TESTO → EMBEDDING (pre-calcolati una volta sola)
# =============================================================================

def get_text_embeddings(clap_wrapper, class_labels: list[str],
                        batch_size: int = 32) -> torch.Tensor:
    """
    Calcola gli embedding testuali per tutte le classi.
    Usa batch_size per non andare OOM con molte classi.
    """
    all_embs = []
    for i in tqdm(range(0, len(class_labels), batch_size),
                  desc="  Encoding text", leave=False):
        batch = class_labels[i:i + batch_size]
        emb   = clap_wrapper.get_text_embeddings(batch)
        all_embs.append(emb.detach().cpu())
    return torch.cat(all_embs, dim=0)   # [n_classes, d_proj]


# =============================================================================
# VALUTAZIONE ZERO-SHOT
# =============================================================================

@torch.no_grad()
def evaluate_zero_shot(clap_wrapper, dataloader, text_embeddings: torch.Tensor,
                        use_cuda: bool, is_residual: bool = False, desc: str = "  Evaluating") -> float:
    """
    Calcola la zero-shot accuracy su un dataloader.
    Funziona sia per CLAP classico sia per ResiDualCLAP.
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    text_emb_norm = F.normalize(text_embeddings.to(device), dim=-1)

    correct = 0
    total   = 0

    for audio, labels in tqdm(dataloader, desc=desc, leave=False):
        audio  = audio.to(device)
        labels = labels.to(device)

        if is_residual:
            if audio.dim() == 3:        # [B, 1, samples] → [B, samples]
                audio = audio.squeeze(1)
            # ResiDualCLAP: passa per audio_encoder
            audio_emb, _ = clap_wrapper.clap.audio_encoder(audio)
        else:
            # CLAP classico
            audio_emb = clap_wrapper._get_audio_embeddings(audio)

        audio_emb_norm = F.normalize(audio_emb, dim=-1)
        logits         = audio_emb_norm @ text_emb_norm.T   # [B, n_classes]
        preds          = logits.argmax(dim=-1)
        correct       += (preds == labels).sum().item()
        total         += labels.size(0)

    return correct / total if total > 0 else 0.0


# =============================================================================
# K-FOLD — ResiDualCLAP
# =============================================================================

def run_kfold_residual(
    clap_wrapper,
    dataset,
    dataset_name:       str,
    residual_config:    dict,
    # K-Fold params
    n_folds:            int   = 5,
    n_samples:          int   = 2000,
    n_pca_samples:      int   = 500,
    seed:               int   = 42,
    # PCA / Lambda optim params
    pca_batch_size:     int   = 16,
    lambda_batch_size:  int   = 16,
    max_epochs:         int   = 30,
    patience:           int   = 5,
    lr:                 float = 1e-2,
    wd:                 float = 1e-2,
    variance_threshold: float = 0.95,
    # Eval params
    eval_batch_size:    int   = 32,
    # Misc
    use_cuda:           bool  = True,
    output_dir:         str   = 'kfolds',
) -> dict:
    """
    K-Fold cross-validation per ResiDualCLAP.

    Per ogni fold:
        1. Split training set in: pca_part | lambda_part
        2. fit_spectral_components su pca_part
        3. optimize_spectral_weights su lambda_part
        4. Valutazione zero-shot sul test fold

    Returns:
        results dict salvato anche su disco in output_dir/<dataset_name>/
    """
    set_seed(seed)

    use_cuda = use_cuda and torch.cuda.is_available()
    clap     = clap_wrapper.clap

    # Campionamento stratificato
    indices  = stratified_sample(dataset, n_samples, seed=seed)
    targets  = np.array([dataset.class_to_idx[dataset.targets[i]] for i in indices])
    classes  = dataset.classes

    # Pre-calcola text embeddings (una volta sola)
    print("\n[ResiDual K-Fold] Calcolo text embeddings...")
    classes = [f"this is the sound of {c.replace('_', ' ')}" for c in dataset.classes]
    text_embs = get_text_embeddings(clap_wrapper, classes)

    # Output directory
    save_dir = Path(output_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {'folds': [], 'dataset': dataset_name, 'type': 'residual',
               'n_folds': n_folds, 'n_samples': len(indices),
               'timestamp': datetime.now().isoformat()}

    for fold_idx, (train_idx_local, test_idx_local) in enumerate(
            tqdm(skf.split(indices, targets), total=n_folds,
                 desc=f"[ResiDual] {dataset_name} K-Fold")):

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        # Indici globali nel dataset originale
        train_global = [indices[i] for i in train_idx_local]
        test_global  = [indices[i] for i in test_idx_local]

        # Split training: pca_part | lambda_part
        set_seed(seed + fold_idx)
        random.shuffle(train_global)
        pca_global    = train_global[:n_pca_samples]
        lambda_global = train_global[n_pca_samples:]

        print(f"  Train: {len(train_global)} | PCA: {len(pca_global)} | "
              f"Lambda: {len(lambda_global)} | Test: {len(test_global)}")

        # DataLoaders
        collate_audio = make_audio_only_collate(clap_wrapper, use_cuda)
        collate_full  = make_collate_fn(clap_wrapper, use_cuda)

        pca_loader    = DataLoader(Subset(dataset, pca_global),
                                   batch_size=pca_batch_size,    shuffle=False,
                                   collate_fn=collate_audio)
        lambda_loader = DataLoader(Subset(dataset, lambda_global),
                                   batch_size=lambda_batch_size,  shuffle=True,
                                   collate_fn=collate_full)
        test_loader   = DataLoader(Subset(dataset, test_global),
                                   batch_size=eval_batch_size,   shuffle=False,
                                   collate_fn=collate_full)

        # 1. Fit PCA
        print(f"\n  [Fold {fold_idx+1}] Fase 1: Fitting PCA...")
        fit_info = clap_wrapper.clap.fit_spectral_components(pca_loader,
                                                              max_samples=n_pca_samples)

        # 2. Ottimizza λ
        print(f"\n  [Fold {fold_idx+1}] Fase 2: Ottimizzazione lambda...")
        text_embs_dev = text_embs.cuda() if use_cuda else text_embs
        history = clap_wrapper.clap.optimize_spectral_weights(
            val_dataloader        = lambda_loader,
            class_text_embeddings = text_embs_dev,
            max_epochs            = max_epochs,
            patience              = patience,
            lr                    = lr,
            wd                    = wd
        )

        # 3. Valutazione zero-shot
        print(f"\n  [Fold {fold_idx+1}] Fase 3: Valutazione zero-shot...")
        accuracy = evaluate_zero_shot(
            clap_wrapper, test_loader, text_embs,
            use_cuda, is_residual=True,
            desc=f"  [Fold {fold_idx+1}/{n_folds}] Evaluating"
        )

        print(f"\n  ✓ Fold {fold_idx+1} accuracy: {accuracy:.4f}")

        fold_result = {
            'fold':              fold_idx + 1,
            'accuracy':          accuracy,
            'best_lambda_epoch': history['best_epoch'],
            'best_lambda_acc':   history['best_accuracy'],
            'lambda_history':    history['accuracy'],
            'n_train':           len(train_global),
            'n_pca':             len(pca_global),
            'n_lambda':          len(lambda_global),
            'n_test':            len(test_global),
        }
        results['folds'].append(fold_result)

    # Statistiche finali
    accs = [f['accuracy'] for f in results['folds']]
    results['mean_accuracy'] = float(np.mean(accs))
    results['std_accuracy']  = float(np.std(accs))
    print(f"\n[ResiDual] {dataset_name} → "
          f"mean={results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    # Salvataggio
    out_path = save_dir / 'residual_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Risultati salvati in {out_path}")

    return results


# =============================================================================
# K-FOLD — CLAP classico
# =============================================================================

def run_kfold_classic(
    clap_wrapper,
    dataset,
    dataset_name:    str,
    # K-Fold params
    n_folds:         int  = 5,
    n_samples:       int  = 2000,
    seed:            int  = 42,
    # Eval params
    eval_batch_size: int  = 32,
    # Misc
    use_cuda:        bool = True,
    output_dir:      str  = 'kfolds',
) -> dict:
    """
    K-Fold cross-validation per CLAP classico (zero-shot, nessuna ottimizzazione).

    Per ogni fold valuta semplicemente l'accuracy zero-shot sul fold di test.

    Returns:
        results dict salvato anche su disco in output_dir/<dataset_name>/
    """
    set_seed(seed)

    use_cuda = use_cuda and torch.cuda.is_available()

    # Campionamento stratificato
    indices = stratified_sample(dataset, n_samples, seed=seed)
    targets = np.array([dataset.class_to_idx[dataset.targets[i]] for i in indices])
    classes = dataset.classes

    # Pre-calcola text embeddings
    print("\n[Classic K-Fold] Calcolo text embeddings...")
    classes = [f"this is the sound of {c.replace('_', ' ')}" for c in dataset.classes]
    text_embs = get_text_embeddings(clap_wrapper, classes)

    # Output directory
    save_dir = Path(output_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {'folds': [], 'dataset': dataset_name, 'type': 'classic',
               'n_folds': n_folds, 'n_samples': len(indices),
               'timestamp': datetime.now().isoformat()}

    collate_full = make_collate_fn(clap_wrapper, use_cuda)

    for fold_idx, (_, test_idx_local) in enumerate(
            tqdm(skf.split(indices, targets), total=n_folds,
                 desc=f"[Classic] {dataset_name} K-Fold")):

        test_global = [indices[i] for i in test_idx_local]
        test_loader = DataLoader(Subset(dataset, test_global),
                                 batch_size=eval_batch_size, shuffle=False,
                                 collate_fn=collate_full)

        accuracy = evaluate_zero_shot(
            clap_wrapper, test_loader, text_embs,
            use_cuda, is_residual=False,
            desc=f"  [Fold {fold_idx+1}/{n_folds}] Evaluating"
        )

        print(f"  Fold {fold_idx+1}/{n_folds} | accuracy: {accuracy:.4f}")

        results['folds'].append({
            'fold':     fold_idx + 1,
            'accuracy': accuracy,
            'n_test':   len(test_global),
        })

    accs = [f['accuracy'] for f in results['folds']]
    results['mean_accuracy'] = float(np.mean(accs))
    results['std_accuracy']  = float(np.std(accs))
    print(f"\n[Classic] {dataset_name} → "
          f"mean={results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    out_path = save_dir / 'classic_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Risultati salvati in {out_path}")

    return results


# =============================================================================
# LOADING RISULTATI
# =============================================================================

def load_all_results(output_dir: str = 'kfolds') -> dict:
    """
    Carica tutti i risultati JSON dalla directory kfolds.

    Returns:
        {dataset_name: {'classic': dict | None, 'residual': dict | None}}
    """
    results = {}
    base = Path(output_dir)

    if not base.exists():
        print(f"Directory {output_dir} non trovata.")
        return results

    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        name = dataset_dir.name
        results[name] = {'classic': None, 'residual': None}

        for variant in ('classic', 'residual'):
            path = dataset_dir / f'{variant}_results.json'
            if path.exists():
                with open(path) as f:
                    results[name][variant] = json.load(f)

    return results
