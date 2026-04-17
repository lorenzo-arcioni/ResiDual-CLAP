"""
residual_kfold.py — K-Fold Validation per ResiDual CLAP e CLAP Classico

Uso tipico:
    from residual_kfold import run_kfold

    results = run_kfold(
        dataset       = esc50_dataset,
        wrapper       = clap_wrapper,        # CLAPWrapper già inizializzato
        class_labels  = esc50_dataset.classes,
        n_folds       = 5,
        pca_samples   = 2000,
        max_epochs    = 20,
        patience      = 5,
        lr            = 1e-2,
        variance_threshold = 0.95,
        target_layers = [1, 2, 3],
        model_type    = 'residual',          # 'residual' o 'classic'
        save_dir      = 'kfolds',
        dataset_name  = 'ESC-50',
    )
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm


# =============================================================================
# Utilità
# =============================================================================

def _build_audio_dataloader(dataset, indices, batch_size=16, shuffle=False):
    """Crea un DataLoader che restituisce solo i path audio (col. 0)."""
    subset = Subset(dataset, indices)

    def collate_paths(batch):
        paths  = [b[0] for b in batch]
        labels = torch.stack([
            torch.tensor(dataset.class_to_idx[b[1]]) for b in batch
        ])
        return paths, labels

    return DataLoader(subset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_paths)


def _compute_accuracy(wrapper, audio_paths, class_labels, batch_size=16):
    """
    Calcola la zero-shot accuracy su una lista di (path, label_str).
    Funziona sia per il wrapper classico che residual (stessa API).
    """
    text_embeddings = wrapper.get_text_embeddings(
        [f"this is the sound of {c}" for c in class_labels]
    )

    all_correct = 0
    all_total   = 0

    for i in tqdm(range(0, len(audio_paths), batch_size),
                  desc="    Valutazione", leave=False):
        batch_paths = audio_paths[i: i + batch_size]
        audio_embs  = wrapper.get_audio_embeddings(batch_paths)
        sims        = wrapper.compute_similarity(audio_embs, text_embeddings)
        preds       = sims.argmax(dim=-1).cpu()

        # Recupera i label corrispondenti a questo batch
        batch_labels = [
            dataset_class_to_idx_lookup[audio_paths[j]]
            for j in range(i, min(i + batch_size, len(audio_paths)))
        ]
        labels_t = torch.tensor(batch_labels)
        all_correct += (preds == labels_t).sum().item()
        all_total   += labels_t.size(0)

    return all_correct / all_total if all_total > 0 else 0.0


# =============================================================================
# Dataloader compatibile con fit_pca_on_data (richiede dict con 'audio')
# =============================================================================

class _AudioOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset leggero che carica waveform grezzi dai path.
    Usato internamente per fit_pca_on_data e optimize_lambda.
    """
    def __init__(self, paths, labels, wrapper, duration=7):
        self.paths    = paths
        self.labels   = labels
        self.wrapper  = wrapper
        self.duration = duration

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio = self.wrapper.load_audio_into_tensor(self.paths[idx], self.duration)
        audio = audio.reshape(1, -1).squeeze(0)
        return audio, self.labels[idx]


def _make_waveform_loader(paths, labels, wrapper, batch_size=8, shuffle=True):
    ds = _AudioOnlyDataset(paths, labels, wrapper)

    def collate(batch):
        wavs   = torch.stack([b[0] for b in batch])
        labels = torch.stack([torch.tensor(b[1]) for b in batch])
        return {'audio': wavs, 'label': labels}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0)


# =============================================================================
# Valutazione zero-shot (senza lookup esterno)
# =============================================================================

def _evaluate_zero_shot(wrapper, paths, labels_int, class_labels, batch_size=16):
    """
    Calcola la zero-shot accuracy.

    Args:
        wrapper:     CLAPWrapper
        paths:       lista di path audio
        labels_int:  lista di interi (indice classe)
        class_labels: lista di stringhe (nomi classi)
        batch_size:  dimensione batch per l'inferenza

    Returns:
        accuracy: float
    """
    prompts = [f"this is the sound of {c}" for c in class_labels]
    text_embs = wrapper.get_text_embeddings(prompts)   # [n_classes, d_proj]

    all_preds  = []
    all_labels = []

    for i in tqdm(range(0, len(paths), batch_size),
                  desc="    Valutazione", leave=False):
        batch_paths  = paths[i: i + batch_size]
        batch_labels = labels_int[i: i + batch_size]

        audio_embs = wrapper.get_audio_embeddings(batch_paths)    # [B, d_proj]
        sims       = wrapper.compute_similarity(audio_embs, text_embs)  # [B, n_classes]
        preds      = sims.argmax(dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    return correct / len(all_labels) if all_labels else 0.0


# =============================================================================
# Funzione principale: run_kfold
# =============================================================================

def run_kfold(
    dataset,
    wrapper,
    class_labels,
    n_folds:            int   = 5,
    pca_fraction:       float = 0.5,   # frazione del training set usata per PCA
    pca_samples:        int   = 2000,  # max campioni per fit_pca
    max_epochs:         int   = 20,
    patience:           int   = 5,
    lr:                 float = 1e-2,
    variance_threshold: float = 0.95,
    target_layers:      list  = None,
    model_type:         str   = 'residual',   # 'residual' o 'classic'
    batch_size_audio:   int   = 8,
    batch_size_eval:    int   = 16,
    save_dir:           str   = 'kfolds',
    dataset_name:       str   = 'dataset',
    seed:               int   = 42,
) -> dict:
    """
    Esegue la k-fold cross-validation su un dataset audio.

    Per ogni fold:
      - Split train / test
      - Train ulteriormente diviso in PCA-split e lambda-split
      - Valuta la baseline (CLAP classico) sul test set
      - Se model_type='residual': fit PCA, ottimizza lambda, valuta ResiDual

    Salva i risultati in `save_dir/<dataset_name>_<model_type>_<timestamp>.json`

    Returns:
        results: dict con accuracy per fold e medie
    """
    if target_layers is None:
        target_layers = [1, 2, 3]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    labels  = [dataset.class_to_idx[t] for t in dataset.targets]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []

    print(f"\n{'='*60}")
    print(f"  K-Fold ({n_folds} fold) — {dataset_name} — tipo: {model_type}")
    print(f"{'='*60}\n")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n{'─'*50}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'─'*50}")
        t0 = time.time()

        # Estrai path e label per train e test
        train_paths  = [dataset.audio_paths[i] for i in train_idx]
        train_labels = [labels[i]              for i in train_idx]
        test_paths   = [dataset.audio_paths[i] for i in test_idx]
        test_labels  = [labels[i]              for i in test_idx]

        # ── Baseline classica ──────────────────────────────────────────
        print("  [1/3] Baseline CLAP classico...")
        baseline_acc = _evaluate_zero_shot(
            wrapper, test_paths, test_labels, class_labels, batch_size_eval
        )
        print(f"        Accuracy baseline: {baseline_acc:.4f}")

        fold_info = {
            'fold':        fold_idx + 1,
            'n_train':     len(train_idx),
            'n_test':      len(test_idx),
            'baseline_acc': baseline_acc,
        }

        # ── ResiDual (solo se richiesto) ───────────────────────────────
        if model_type == 'residual':

            # Divide il training set in PCA-split e lambda-split
            n_pca = int(len(train_paths) * pca_fraction)
            pca_paths    = train_paths[:n_pca]
            pca_labels   = train_labels[:n_pca]
            lam_paths    = train_paths[n_pca:]
            lam_labels   = train_labels[n_pca:]

            print(f"  [2/3] Fit PCA ({len(pca_paths)} campioni, max={pca_samples})...")

            # Riconfigura il modello con i parametri scelti
            clap_model = wrapper.clap
            htsat      = clap_model.audio_encoder.base.htsat

            # Aggiorna la configurazione residual in-place
            htsat.target_layers      = target_layers
            htsat.variance_threshold = variance_threshold
            htsat._build_spectral_layers()

            # DataLoader per fit_pca
            pca_loader = _make_waveform_loader(
                pca_paths, pca_labels, wrapper,
                batch_size=batch_size_audio, shuffle=True
            )
            fit_info = clap_model.audio_encoder.base.htsat.fit_pca_on_data(
                pca_loader, max_samples=pca_samples
            )

            print(f"  [3/3] Ottimizzazione lambda ({len(lam_paths)} campioni)...")

            # Pre-calcola text embeddings per le classi
            prompts    = [f"this is the sound of {c}" for c in class_labels]
            text_embs  = wrapper.get_text_embeddings(prompts)   # [n_classes, d_proj]

            # DataLoader per optimize_lambda
            lam_loader = _make_waveform_loader(
                lam_paths, lam_labels, wrapper,
                batch_size=batch_size_audio, shuffle=True
            )

            history = clap_model.audio_encoder.base.htsat.optimize_lambda(
                val_dataloader        = lam_loader,
                class_text_embeddings = text_embs,
                projection            = clap_model.audio_encoder.projection,
                max_epochs            = max_epochs,
                patience              = patience,
                lr                    = lr,
            )

            print("  Valutazione ResiDual ottimizzato...")
            residual_acc = _evaluate_zero_shot(
                wrapper, test_paths, test_labels, class_labels, batch_size_eval
            )
            print(f"        Accuracy ResiDual: {residual_acc:.4f}")

            fold_info['residual_acc']   = residual_acc
            fold_info['lambda_history'] = history
            fold_info['fit_info']       = {
                k: str(v) for k, v in fit_info.items()
            }

        elapsed = time.time() - t0
        fold_info['elapsed_sec'] = elapsed
        fold_results.append(fold_info)

        print(f"  Fold completato in {elapsed:.1f}s")

    # ── Riepilogo ──────────────────────────────────────────────────────
    baseline_accs = [f['baseline_acc'] for f in fold_results]
    summary = {
        'dataset':          dataset_name,
        'model_type':       model_type,
        'n_folds':          n_folds,
        'class_labels':     class_labels,
        'config': {
            'target_layers':      target_layers,
            'variance_threshold': variance_threshold,
            'pca_fraction':       pca_fraction,
            'pca_samples':        pca_samples,
            'max_epochs':         max_epochs,
            'patience':           patience,
            'lr':                 lr,
        },
        'folds':               fold_results,
        'mean_baseline_acc':   float(np.mean(baseline_accs)),
        'std_baseline_acc':    float(np.std(baseline_accs)),
    }

    if model_type == 'residual':
        residual_accs = [f['residual_acc'] for f in fold_results]
        summary['mean_residual_acc'] = float(np.mean(residual_accs))
        summary['std_residual_acc']  = float(np.std(residual_accs))

    # ── Salvataggio ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f"{dataset_name}_{model_type}_{timestamp}.json"
    out_path  = save_path / filename

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Risultati salvati in: {out_path}")
    print(f"  Baseline media:  {summary['mean_baseline_acc']:.4f} ± {summary['std_baseline_acc']:.4f}")
    if model_type == 'residual':
        print(f"  ResiDual media:  {summary['mean_residual_acc']:.4f} ± {summary['std_residual_acc']:.4f}")
    print(f"{'='*60}\n")

    return summary


# =============================================================================
# Caricamento e analisi risultati
# =============================================================================

def load_all_results(kfold_dir: str = 'kfolds') -> list[dict]:
    """
    Carica tutti i file JSON nella cartella kfolds.
    Raggruppa per (dataset, model_type) e restituisce l'ultimo run.
    """
    kfold_path = Path(kfold_dir)
    all_files  = sorted(kfold_path.glob('*.json'))

    results = []
    for fp in all_files:
        with open(fp) as f:
            results.append(json.load(f))

    return results


def summarize_results(results: list[dict]) -> dict:
    """
    Dato l'output di load_all_results, produce un dizionario:
        {dataset_name: {'classic': acc, 'residual': acc}}
    Prende l'ultimo risultato per ogni (dataset, tipo).
    """
    summary = {}
    for r in results:
        ds   = r['dataset']
        mtype = r['model_type']
        if ds not in summary:
            summary[ds] = {}

        entry = {'mean': r['mean_baseline_acc'], 'std': r['std_baseline_acc']}
        summary[ds]['classic'] = entry

        if mtype == 'residual' and 'mean_residual_acc' in r:
            summary[ds]['residual'] = {
                'mean': r['mean_residual_acc'],
                'std':  r['std_residual_acc'],
            }

    return summary
