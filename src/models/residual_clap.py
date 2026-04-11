"""
ResiDual HTSAT — Spectral Reweighting con Pesi Lambda Learnable (RD*)

Implementazione (Eq. 6):
    RD(X, λ) = Φ⁻¹ · diag(λ) · Φ · (X - μ)

dove:
    Φ, μ  : basi PCA fisse, calcolate una volta su dati audio (non aggiornate)
    λ      : vettore di pesi learnable, uno per ogni PC, ottimizzato su val set

Struttura:
    1. SpectralReweightingLayer   — RD(X, λ) con λ come nn.Parameter
    2. AttentionHook              — intercetta heads via forward hook
    3. ResiDualHTSAT              — modello principale
    4. Wrappers                   — ResiDualHTSATWrapper, AudioEncoder, ResiDualCLAP
"""

import math
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Dict, List, Optional

from .clap import CLAP, Projection
from .htsat import HTSAT_Swin_Transformer, WindowAttention
from . import config as htsat_config


# =============================================================================
# 1. SPECTRAL REWEIGHTING LAYER — RD(X, λ)
# =============================================================================

class SpectralReweightingLayer(nn.Module):
    """
    Applica la trasformazione ResiDual (Eq. 6):

        RD(X, λ) = Φ⁻¹ · diag(λ) · Φ · (X - μ)

    Φ e μ sono fissati da fit_pca e non vengono aggiornati.
    λ è un nn.Parameter ottimizzato via gradient descent.

    Inizializzazione di λ dopo fit_pca:
        λ_j = 1.0  per j <= k   (PC che spiegano variance_threshold della varianza)
        λ_j = 0.0  per j >  k   (PC rumore)

    Durante l'ottimizzazione λ può assumere qualsiasi valore reale:
        λ_j > 1  : amplifica la componente j
        0 < λ_j < 1 : attenua
        λ_j = 0  : sopprime
        λ_j < 0  : inverte
    """

    def __init__(self, embed_dim: int, variance_threshold: float = 0.95):
        """
        Args:
            embed_dim:          Dimensione dell'head (head_dim = layer_dim // num_heads)
            variance_threshold: Soglia per determinare k empiricamente (0.95 o 0.90)
        """
        super().__init__()
        self.embed_dim          = embed_dim
        self.variance_threshold = variance_threshold

        # Basi PCA: fisse dopo fit_pca, non aggiornate dall'ottimizzatore
        self.register_buffer('pca_components', torch.eye(embed_dim))   # Φ: [embed_dim, embed_dim]
        self.register_buffer('pca_mean',       torch.zeros(embed_dim)) # μ: [embed_dim]
        self.register_buffer('is_fitted',      torch.tensor(False))

        # λ: learnable, inizializzato a zeros — verrà impostato da fit_pca
        # Definito come Parameter subito per essere trovato dall'ottimizzatore
        self.lambda_weights = nn.Parameter(torch.zeros(embed_dim))

    def fit_pca(self, data: torch.Tensor) -> Dict:
        """
        Calcola le basi PCA (Φ, μ) e inizializza λ.

        Le basi vengono fissate e non aggiornate.
        λ viene inizializzato a 1 per le prime k PC e 0 per le restanti.

        Args:
            data: [N_samples, embed_dim] — hidden states raccolti

        Returns:
            info: k, varianza spiegata, profilo eigenvalori per logging/analisi
        """
        X = data.detach().cpu().numpy()

        pca = PCA(n_components=min(X.shape))
        pca.fit(X)

        evr        = pca.explained_variance_ratio_
        evr_cumsum = np.cumsum(evr)

        # k empirico per-testa: minimo j tale che EVR_cumsum[j] >= variance_threshold
        k = int(np.argmax(evr_cumsum >= self.variance_threshold) + 1)
        k = max(1, min(k, self.embed_dim))

        # Fissa le basi PCA (non verranno toccate dall'ottimizzatore)
        self.pca_components.data = torch.tensor(
            pca.components_.T, dtype=torch.float32   # [embed_dim, n_components]
        )
        self.pca_mean.data = torch.tensor(pca.mean_, dtype=torch.float32)

        # Inizializza λ: 1 per le k PC task-relevant, 0 per il rumore
        lambda_init        = torch.zeros(pca.components_.shape[0])
        lambda_init[:k]    = 1.0
        self.lambda_weights = nn.Parameter(lambda_init)

        self.is_fitted.data = torch.tensor(True)

        return {
            'k':                  k,
            'variance_at_k':      float(evr_cumsum[k - 1]),
            'evr_pc1':            float(evr[0]),
            'evr_top3':           float(evr_cumsum[min(2, len(evr) - 1)]),
            'eigenvalue_profile': evr.tolist(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RD(X, λ) = Φ⁻¹ · diag(λ) · Φ · (X - μ)

        Con Φ ortogonale: Φ⁻¹ = Φᵀ → ricostruzione = proj_weighted @ Φᵀ

        Args:
            x: [..., embed_dim]

        Returns:
            x_reweighted: stessa shape di x
        """
        if not self.is_fitted:
            return x

        shape  = x.shape
        x_flat = x.reshape(-1, self.embed_dim)

        x_centered    = x_flat - self.pca_mean                    # (X - μ)
        proj          = x_centered @ self.pca_components          # Φ · (X - μ)ᵀ  → [N, n_components]
        weighted_proj = proj * self.lambda_weights                 # diag(λ) · proj
        reconstructed = weighted_proj @ self.pca_components.T     # Φ⁻¹ · weighted

        return reconstructed.reshape(shape)
        # Nota: non riaggiungo μ perché l'output viene passato alla projection
        # del blocco attention che ha il proprio bias; riaggiungerla causerebbe
        # uno shift sistematico non desiderato.


# =============================================================================
# 2. ATTENTION HOOK
# =============================================================================

class AttentionHook:
    """
    Hook su WindowAttention per raccolta dati o applicazione reweighting.

    Installato temporaneamente da _forward_layer_with_attention_hooks,
    rimosso subito dopo — nessuna modifica permanente al modello.
    """

    def __init__(self, spectral_layers: nn.ModuleList, num_heads: int,
                 collected_data: dict, layer_idx: int, block_idx: int,
                 collect_for_fitting: bool):
        self.spectral_layers     = spectral_layers
        self.num_heads           = num_heads
        self.collected_data      = collected_data
        self.layer_idx           = layer_idx
        self.block_idx           = block_idx
        self.collect_for_fitting = collect_for_fitting

    def __call__(self, module: WindowAttention, inputs, outputs):
        x_out, attn = outputs
        B_, N, C    = x_out.shape
        head_dim    = C // self.num_heads

        # Ricalcola V dall'input (stessi pesi, no grad aggiuntivo)
        x_in = inputs[0]
        qkv  = module.qkv(x_in).reshape(B_, N, 3, self.num_heads, head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        v    = qkv[2]                                       # [B_, n_heads, N, head_dim]

        x_heads = attn @ v                                  # [B_, n_heads, N, head_dim]

        if self.collect_for_fitting:
            self._collect(x_heads, head_dim)
            return outputs

        # Applica RD(X, λ) per ogni head
        reweighted = torch.stack([
            self.spectral_layers[h](x_heads[:, h])
            for h in range(self.num_heads)
        ], dim=1)                                           # [B_, n_heads, N, head_dim]

        x_reweighted = reweighted.transpose(1, 2).reshape(B_, N, C)
        x_reweighted = module.proj(x_reweighted)
        x_reweighted = module.proj_drop(x_reweighted)

        return x_reweighted, attn

    def _collect(self, x_heads: torch.Tensor, head_dim: int):
        layer_key = f'layer_{self.layer_idx}'
        block_key = self.block_idx

        self.collected_data.setdefault(layer_key, {})
        self.collected_data[layer_key].setdefault(block_key, {
            f'head_{h}': [] for h in range(self.num_heads)
        })

        for h in range(self.num_heads):
            head_data = x_heads[:, h].reshape(-1, head_dim)
            self.collected_data[layer_key][block_key][f'head_{h}'].append(
                head_data.detach().cpu()
            )


# =============================================================================
# 3. RESIDUAL HTSAT v3
# =============================================================================

class ResiDualHTSAT(HTSAT_Swin_Transformer):
    """
    HTSAT con RD* spectral reweighting.

    Config minimale:
        residual_config = {
            'target_layers':      List[int],   # es. [1, 2, 3] — scelti da analisi id_df
            'variance_threshold': float,       # 0.95 (pca_95) o 0.90 (pca_90)
        }

    Pipeline completa:
        1. fit_pca_on_data(dataloader)    → calcola Φ, μ, inizializza λ per ogni head
        2. optimize_lambda(val_loader,    → ottimizza λ via gradient descent
                           text_embeddings,     su zero-shot accuracy
                           text_encoder)
        3. forward(audio)                 → inference con λ ottimizzati
    """

    def __init__(self, *args, residual_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual_config    = residual_config
        self.target_layers      = residual_config.get('target_layers', [1, 2, 3])
        self.variance_threshold = residual_config.get('variance_threshold', 0.95)

        self.spectral_layers = nn.ModuleDict()
        self.collected_data: Dict = {}

        self._build_spectral_layers()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _build_spectral_layers(self):
        for layer_idx in self.target_layers:
            if layer_idx >= len(self.layers):
                print(f"[ResiDual] Layer {layer_idx} non esiste, skip")
                continue

            layer_dim  = int(self.embed_dim * (2 ** layer_idx))
            num_heads  = self.num_heads[layer_idx]
            num_blocks = self.depths[layer_idx]
            head_dim   = layer_dim // num_heads
            layer_name = f'layer_{layer_idx}'

            self.spectral_layers[layer_name] = nn.ModuleList([
                nn.ModuleList([
                    SpectralReweightingLayer(
                        embed_dim          = head_dim,
                        variance_threshold = self.variance_threshold,
                    )
                    for _ in range(num_heads)
                ])
                for _ in range(num_blocks)
            ])

    # -------------------------------------------------------------------------
    # Forward (identico a v2)
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False,
                collect_for_fitting: bool = False):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            from .pytorch_utils import do_mixup
            x = do_mixup(x, mixup_lambda)

        if infer_mode:
            x = self._prepare_infer(x)
            return self._forward_features(x, collect_for_fitting)

        if self.config.enable_repeat_mode:
            return self._forward_repeat_mode(x, collect_for_fitting)

        return self._forward_standard_mode(x, collect_for_fitting)

    def _prepare_infer(self, x):
        target_T     = int(self.spec_size * self.freq_ratio)
        repeat_ratio = math.floor(target_T / x.shape[2])
        return self.reshape_wav2img(x.repeat(1, 1, repeat_ratio, 1))

    def _forward_repeat_mode(self, x, collect_for_fitting):
        if self.training:
            cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
            return self._forward_features(self.repeat_wat2img(x, cur_pos), collect_for_fitting)
        outputs = [
            self._forward_features(self.repeat_wat2img(x.clone(), p), False)
            for p in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size)
        ]
        return self._average_outputs(outputs)

    def _forward_standard_mode(self, x, collect_for_fitting):
        max_len = self.freq_ratio * self.spec_size
        if x.shape[2] <= max_len:
            return self._forward_features(self.reshape_wav2img(x), collect_for_fitting)
        if self.training:
            return self._forward_features(
                self.reshape_wav2img(self.crop_wav(x, crop_size=max_len)),
                collect_for_fitting
            )
        overlap, crop = 344, 689
        outputs = [
            self._forward_features(
                self.reshape_wav2img(self.crop_wav(x, crop, p)), False
            )
            for p in range(0, x.shape[2] - crop - 1, overlap)
        ]
        return self._average_outputs(outputs)

    def _forward_features(self, x, collect_for_fitting):
        frames_num = x.shape[2]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        attn = None
        for i, layer in enumerate(self.layers):
            x, attn = self._forward_layer(i, layer, x, collect_for_fitting)

        return self._finalize_output(x, attn, frames_num)

    def _forward_layer(self, layer_idx, layer, x, collect_for_fitting):
        layer_name = f'layer_{layer_idx}'
        if layer_name not in self.spectral_layers:
            return layer(x)
        return self._forward_layer_with_attention_hooks(
            layer, layer_idx, x, collect_for_fitting
        )

    def _forward_layer_with_attention_hooks(self, layer, layer_idx, x, collect_for_fitting):
        layer_name     = f'layer_{layer_idx}'
        spectral_block = self.spectral_layers[layer_name]
        handles        = []

        for block_idx, block in enumerate(layer.blocks):
            hook = AttentionHook(
                spectral_layers     = spectral_block[block_idx],
                num_heads           = block.attn.num_heads,
                collected_data      = self.collected_data,
                layer_idx           = layer_idx,
                block_idx           = block_idx,
                collect_for_fitting = collect_for_fitting,
            )
            handles.append(block.attn.register_forward_hook(hook))

        x, attn = layer(x)

        for h in handles:
            h.remove()

        return x, attn

    # -------------------------------------------------------------------------
    # Finalize output (identico a v2)
    # -------------------------------------------------------------------------

    def _finalize_output(self, x, attn, frames_num):
        from .pytorch_utils import interpolate

        if self.config.enable_tscam:
            x = self.norm(x)
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x  = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape
            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            latent_output = self.avgpool(torch.flatten(x, 2))
            latent_output = torch.flatten(latent_output, 1)

            if self.config.htsat_attn_heatmap and attn is not None:
                attn = self._process_attention_heatmap(attn, B, SF, ST)

            x   = self.tscam_conv(x)
            x   = torch.flatten(x, 2)
            fpx = torch.sigmoid(x).permute(0, 2, 1).contiguous()
            if self.config.htsat_attn_heatmap and attn is not None:
                fpx = fpx * attn
            fpx = interpolate(fpx, 8 * self.patch_stride[1])
            x   = self.avgpool(x)
            x   = torch.flatten(x, 1)

            clipwise = x if self.config.loss_type == "clip_ce" else torch.sigmoid(x)
            return {'framewise_output': fpx, 'clipwise_output': clipwise,
                    'latent_output': latent_output}

        else:
            x = self.norm(x)
            B, N, C = x.shape
            side = frames_num // (2 ** (len(self.depths) + 1))
            fpx  = x.permute(0, 2, 1).contiguous().reshape(B, C, side, side)
            B, C, F, T = fpx.shape
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim=2)
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            x   = self.avgpool(x.transpose(1, 2))
            x   = torch.flatten(x, 1)
            if self.num_classes > 0:
                x   = self.head(x)
                fpx = self.head(fpx)
            return {'framewise_output': torch.sigmoid(fpx),
                    'clipwise_output':  torch.sigmoid(x)}

    def _process_attention_heatmap(self, attn, B, SF, ST):
        attn       = torch.mean(attn, dim=1).mean(dim=1).reshape(B, SF, ST)
        c_freq_bin = SF // self.freq_ratio
        attn       = attn.reshape(B, SF // c_freq_bin, c_freq_bin, ST)
        attn       = attn.permute(0, 2, 1, 3).contiguous().reshape(B, c_freq_bin, -1)
        attn       = attn.mean(dim=1)
        attn_max   = torch.max(attn, dim=1, keepdim=True)[0]
        attn_min   = torch.min(attn, dim=1, keepdim=True)[0]
        attn       = ((attn * 0.15) + (attn_max * 0.85 - attn_min)) / (attn_max - attn_min)
        return attn.unsqueeze(dim=2)

    def _average_outputs(self, output_dicts):
        return {k: torch.stack([d[k] for d in output_dicts]).mean(0)
                for k in output_dicts[0]}

    # -------------------------------------------------------------------------
    # Fase 1: fit PCA (calcola Φ, μ, inizializza λ)
    # -------------------------------------------------------------------------

    def fit_pca_on_data(self, dataloader, max_samples: int = 10000) -> Dict:
        """
        Raccoglie hidden states e calcola le basi PCA per ogni head target.

        Dopo questa chiamata:
          - Φ e μ sono fissati in ogni SpectralReweightingLayer
          - λ è inizializzato: 1 per le prime k PC, 0 per le restanti
          - Il modello è pronto per optimize_lambda

        Args:
            dataloader:  DataLoader con audio (stesso split usato per l'analisi)
            max_samples: Numero massimo di campioni da raccogliere

        Returns:
            fit_info: Per ogni head → k, varianza spiegata, profilo eigenvalori
        """
        self.eval()
        self.collected_data.clear()

        # Raccolta hidden states
        n_samples = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[ResiDual] Raccolta per PCA"):
                if n_samples >= max_samples:
                    break
                audio = self._extract_audio(batch)
                if next(self.parameters()).is_cuda and not audio.is_cuda:
                    audio = audio.cuda()
                try:
                    self.forward(audio, collect_for_fitting=True)
                    n_samples += audio.size(0)
                except Exception as e:
                    print(f"[ResiDual] Errore: {e}")

        print(f"[ResiDual] Raccolti {n_samples} campioni")

        # Fit PCA per ogni head, inizializza λ
        fit_info = {}
        for layer_name, block_data in self.collected_data.items():
            fit_info[layer_name] = {}
            block_spectral = self.spectral_layers[layer_name]

            for block_idx, head_data in block_data.items():
                fit_info[layer_name][f'block_{block_idx}'] = {}

                for head_name, data_list in head_data.items():
                    if not data_list:
                        continue
                    combined   = torch.cat(data_list, dim=0)
                    head_idx   = int(head_name.split('_')[1])
                    info       = block_spectral[block_idx][head_idx].fit_pca(combined)
                    fit_info[layer_name][f'block_{block_idx}'][head_name] = info

                    print(f"  {layer_name} | block {block_idx} | {head_name}: "
                          f"k={info['k']} | var@k={info['variance_at_k']:.3f}")

        self.collected_data.clear()
        return fit_info

    # -------------------------------------------------------------------------
    # Fase 2: ottimizzazione di λ su zero-shot accuracy
    # -------------------------------------------------------------------------

    def optimize_lambda(self,
                        val_dataloader,
                        class_text_embeddings: torch.Tensor,
                        projection: nn.Module,
                        max_epochs:  int   = 30,
                        patience:    int   = 5,
                        lr:          float = 1e-2) -> Dict:
        """
        Ottimizza i pesi λ di ogni SpectralReweightingLayer per massimizzare
        la zero-shot accuracy sul validation set.

        Solo i λ vengono aggiornati. Φ, μ e tutti gli altri parametri
        del modello sono congelati.

        Args:
            val_dataloader:        DataLoader con (audio, label) del validation set
            class_text_embeddings: [n_classes, d_proj] — text embeddings pre-calcolati
                                   per ogni classe (es. "the sound of a dog")
            max_epochs:            Numero massimo di epoche (default: 30)
            patience:              Early stopping patience (default: 5)
            lr:                    Learning rate per Schedule-Free Adam (default: 1e-2)

        Returns:
            history: {'accuracy': [...], 'best_epoch': int, 'best_accuracy': float}
        """
        device = next(self.parameters()).device

        # Congela tutto tranne i lambda
        self._freeze_all_except_lambda()

        # Raccoglie solo i parametri lambda da ottimizzare
        lambda_params = self._get_lambda_parameters()
        if not lambda_params:
            raise RuntimeError("[ResiDual] Nessun lambda trovato. Hai chiamato fit_pca_on_data?")

        # Schedule-Free Adam (approssimato con Adam + cosine annealing)
        # Per Schedule-Free Adam vero: pip install schedulefree
        try:
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFree(lambda_params, lr=lr)
        except ImportError:
            print("[ResiDual] schedulefree non installato, uso Adam standard")
            optimizer = torch.optim.Adam(lambda_params, lr=lr)

        class_text_embeddings = class_text_embeddings.to(device)
        class_text_embeddings = nn.functional.normalize(class_text_embeddings, dim=-1)

        best_accuracy  = -1.0
        best_lambda    = self._snapshot_lambda()
        epochs_no_imp  = 0
        history        = {'accuracy': [], 'best_epoch': 0, 'best_accuracy': 0.0}

        for epoch in range(max_epochs):
            # Training step: calcola accuracy e backprop su λ
            epoch_accuracy = self._optimize_epoch(
                val_dataloader, class_text_embeddings, optimizer, device, projection
            )
            history['accuracy'].append(epoch_accuracy)

            print(f"  Epoch {epoch+1:3d}/{max_epochs} | acc={epoch_accuracy:.4f} "
                  f"| best={best_accuracy:.4f}")

            # Early stopping
            if epoch_accuracy > best_accuracy:
                best_accuracy       = epoch_accuracy
                best_lambda         = self._snapshot_lambda()
                epochs_no_imp       = 0
                history['best_epoch']    = epoch + 1
                history['best_accuracy'] = best_accuracy
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"[ResiDual] Early stopping a epoca {epoch+1}")
                    break

        # Ripristina i migliori lambda trovati
        self._restore_lambda(best_lambda)
        self._unfreeze_all()

        print(f"[ResiDual] Ottimizzazione completata. "
              f"Best accuracy: {best_accuracy:.4f} @ epoch {history['best_epoch']}")

        return history

    def _optimize_epoch(self, dataloader, class_text_embeddings, optimizer,
                        device, projection: nn.Module) -> float:
        """
        Un'epoca di ottimizzazione: calcola zero-shot accuracy e backpropaga su λ.

        Il gradiente fluisce:
            cross_entropy → logits → audio_emb_proj → projection(latent_output)
            → latent_output → SpectralReweightingLayer → λ

        La projection è congelata (solo λ ha requires_grad=True),
        ma deve essere nel path del gradiente per collegare
        latent_output (768) allo spazio text (1024).
        """
        self.eval()

        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            audio, labels = self._extract_audio_and_labels(batch)
            audio  = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward: gradiente abilitato solo per λ (tutto il resto è congelato)
            out           = self.forward(audio)
            latent        = out['latent_output']                     # [B, 768]

            # Proietta nello spazio CLAP condiviso con i text embeddings [B, 1024]
            audio_emb_proj = projection(latent)
            audio_emb_norm = nn.functional.normalize(audio_emb_proj, dim=-1)

            # Similarità coseno con i text embeddings delle classi
            logits = audio_emb_norm @ class_text_embeddings.T       # [B, n_classes]

            # Cross-entropy: minimizzare ≡ massimizzare zero-shot accuracy
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            preds          = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        return total_correct / total_samples if total_samples > 0 else 0.0

    # -------------------------------------------------------------------------
    # Utilities per gestione lambda
    # -------------------------------------------------------------------------

    def _get_lambda_parameters(self) -> List[nn.Parameter]:
        """Restituisce tutti i parametri λ degli spectral layers."""
        return [
            layer.lambda_weights
            for layer_modules in self.spectral_layers.values()
            for block_modules in layer_modules
            for layer in block_modules
            if layer.is_fitted
        ]

    def _freeze_all_except_lambda(self):
        """Congela tutti i parametri tranne i λ degli spectral layers."""
        lambda_ids = {id(p) for p in self._get_lambda_parameters()}
        for p in self.parameters():
            if id(p) not in lambda_ids:
                p.requires_grad_(False)

    def _unfreeze_all(self):
        """Ripristina requires_grad per tutti i parametri."""
        for p in self.parameters():
            p.requires_grad_(True)

    def _snapshot_lambda(self) -> Dict[str, torch.Tensor]:
        """Salva una copia dei λ correnti."""
        return {
            f'{ln}_{bi}_{hi}': layer.lambda_weights.detach().clone()
            for ln, layer_modules in self.spectral_layers.items()
            for bi, block_modules in enumerate(layer_modules)
            for hi, layer in enumerate(block_modules)
        }

    def _restore_lambda(self, snapshot: Dict[str, torch.Tensor]):
        """Ripristina i λ da uno snapshot."""
        for ln, layer_modules in self.spectral_layers.items():
            for bi, block_modules in enumerate(layer_modules):
                for hi, layer in enumerate(block_modules):
                    key = f'{ln}_{bi}_{hi}'
                    if key in snapshot:
                        with torch.no_grad():
                            layer.lambda_weights.copy_(snapshot[key])

    @staticmethod
    def _extract_audio(batch) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch.get('audio', batch.get('waveform'))
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    @staticmethod
    def _extract_audio_and_labels(batch):
        if isinstance(batch, dict):
            return batch.get('audio', batch.get('waveform')), batch['label']
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        raise ValueError("Formato batch non riconosciuto")


# =============================================================================
# 4. WRAPPERS
# =============================================================================

class ResiDualHTSATWrapper(nn.Module):
    def __init__(self, residual_config: Dict):
        super().__init__()
        self.htsat = ResiDualHTSAT(config=htsat_config, residual_config=residual_config)

    def forward(self, x):
        out = self.htsat(x)
        out['embedding'] = out['latent_output']
        return out


class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name: str, d_in: int, d_out: int,
                 sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int,
                 classes_num: int, residual_config: Dict):
        super().__init__()
        self.base       = ResiDualHTSATWrapper(residual_config=residual_config)
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out = self.base(x)
        return self.projection(out['embedding']), out['clipwise_output']


class ResiDualCLAP(CLAP):
    """
    CLAP con RD* spectral reweighting.

    Uso tipico:

        # 1. Costruisci e carica pesi
        model = ResiDualCLAP(..., residual_config={
            'target_layers':      [1, 2, 3],
            'variance_threshold': 0.95,
        })
        model.load_state_dict(clap_weights)

        # 2. Calcola basi PCA e inizializza lambda
        fit_info = model.fit_spectral_components(train_loader)

        # 3. Pre-calcola text embeddings per le classi
        text_embs = model.encode_text(class_prompts)   # [n_classes, d_proj]

        # 4. Ottimizza lambda su zero-shot accuracy
        history = model.optimize_spectral_weights(val_loader, text_embs)

        # 5. Inference
        audio_emb, _ = model.audio_encoder(audio)

    Config per dataset generico (ESC-50):
        target_layers=[1,2,3], variance_threshold=0.95

    Config per dataset specifico (VocalSound):
        target_layers=[2,3], variance_threshold=0.90
    """

    def __init__(self, *args, residual_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_encoder = AudioEncoder(
            kwargs["audioenc_name"], kwargs["out_emb"],    kwargs["d_proj"],
            kwargs["sample_rate"],   kwargs["window_size"], kwargs["hop_size"],
            kwargs["mel_bins"],      kwargs["fmin"],        kwargs["fmax"],
            kwargs["classes_num"],   residual_config,
        )

    def fit_spectral_components(self, audio_dataloader,
                                max_samples: int = 10000) -> Dict:
        """Fase 1: calcola Φ, μ e inizializza λ per ogni head target."""
        return self.audio_encoder.base.htsat.fit_pca_on_data(
            audio_dataloader, max_samples
        )

    def optimize_spectral_weights(self,
                                  val_dataloader,
                                  class_text_embeddings: torch.Tensor,
                                  max_epochs: int   = 30,
                                  patience:   int   = 5,
                                  lr:         float = 1e-2) -> Dict:
        """
        Fase 2: ottimizza λ su zero-shot accuracy.

        Passa la projection layer di AudioEncoder a optimize_lambda
        in modo che il gradiente possa fluire correttamente:
            latent_output (768) → projection → spazio CLAP (1024) → similarity con text
        """
        projection = self.audio_encoder.projection  # Projection(768 → 1024), congelata
        return self.audio_encoder.base.htsat.optimize_lambda(
            val_dataloader,
            class_text_embeddings,
            projection  = projection,
            max_epochs  = max_epochs,
            patience    = patience,
            lr          = lr,
        )