"""
ResiDual HTSAT — Spectral Reweighting con Pesi Lambda Learnable (RD*)

Implementazione (Eq. 6):
    RD(X, λ) = Φ⁻¹ · diag(λ) · Φ · (X - μ)

dove:
    Φ, μ  : basi PCA fisse, calcolate una volta su dati audio (non aggiornate)
    λ      : vettore di pesi learnable, uno per ogni PC, ottimizzato su val set

Struttura:
    1. SpectralReweightingLayer   — RD(X, λ) con λ come nn.Parameter
    2. AttentionHook              — hook permanente su WindowAttention
    3. ResiDualHTSAT              — modello principale
    4. Wrappers                   — ResiDualHTSATWrapper, AudioEncoder, ResiDualCLAP
"""

import math
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from typing import Dict, List

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
        λ_j > 1      : amplifica la componente j
        0 < λ_j < 1  : attenua
        λ_j = 0      : sopprime
        λ_j < 0      : inverte
    """

    def __init__(self, embed_dim: int, variance_threshold: float = 0.95):
        super().__init__()
        self.embed_dim          = embed_dim
        self.variance_threshold = variance_threshold

        # Basi PCA: fisse dopo fit_pca, non aggiornate dall'ottimizzatore
        self.register_buffer('pca_components', torch.eye(embed_dim))    # Φ  [D, D]
        self.register_buffer('pca_mean',       torch.zeros(embed_dim))  # μ  [D]
        self.register_buffer('is_fitted',      torch.tensor(False))

        # λ: learnable, inizializzato a zeros — verrà impostato da fit_pca
        self.lambda_weights = nn.Parameter(torch.zeros(embed_dim))

    def fit_pca(self, data: torch.Tensor) -> Dict:
        """
        Calcola le basi PCA (Φ, μ) e inizializza λ.

        Args:
            data: [N_samples, embed_dim] — hidden states raccolti

        Returns:
            info dict: k, varianza spiegata, profilo eigenvalori
        """
        device = self.pca_components.device  # ← leggi device prima di sovrascrivere

        X   = data.detach().cpu().numpy()
        pca = PCA(n_components=min(X.shape))
        pca.fit(X)

        evr        = pca.explained_variance_ratio_
        evr_cumsum = np.cumsum(evr)

        k = int(np.argmax(evr_cumsum >= self.variance_threshold) + 1)
        k = max(1, min(k, self.embed_dim))

        self.pca_components.data = torch.tensor(
            pca.components_.T, dtype=torch.float32
        ).to(device)

        self.pca_mean.data = torch.tensor(
            pca.mean_, dtype=torch.float32
        ).to(device)

        lambda_init     = torch.zeros(pca.components_.shape[0], device=device)
        lambda_init[:k] = 1.0
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

        Con Φ ortogonale: Φ⁻¹ = Φᵀ

        Args:
            x: [..., embed_dim]

        Returns:
            x_reweighted: stessa shape di x
        """
        if not self.is_fitted:
            return x

        shape         = x.shape
        x_flat        = x.reshape(-1, self.embed_dim)
        proj          = (x_flat - self.pca_mean) @ self.pca_components        # Φ·(X-μ)
        reconstructed = (proj * self.lambda_weights) @ self.pca_components.T  # Φᵀ·diag(λ)·proj

        return reconstructed.reshape(shape)
        # Nota: μ non viene riaggiunte — l'output projection del blocco attention
        # ha il proprio bias che assorbe lo shift sistematico.


# =============================================================================
# 2. ATTENTION HOOK (permanente, con flag collect_for_fitting)
# =============================================================================

class AttentionHook:
    """
    Hook permanente su WindowAttention.

    Opera in due modalità controllate da self.collect_for_fitting:
      - collection mode  : accumula gli hidden states per head senza modificare
                           l'output del forward, alimentando fit_pca_on_data.
      - reweighting mode : applica RD* per ogni head e sostituisce l'output
                           di WindowAttention prima della output projection.

    Non ha stato proprio oltre al riferimento agli SpectralReweightingLayer
    e al dizionario di raccolta dati — entrambi condivisi con ResiDualHTSAT.
    """

    def __init__(self, spectral_layers: nn.ModuleList, num_heads: int,
                 collected_data: dict, layer_idx: int, block_idx: int):
        self.spectral_layers     = spectral_layers
        self.num_heads           = num_heads
        self.collected_data      = collected_data
        self.layer_idx           = layer_idx
        self.block_idx           = block_idx
        self.collect_for_fitting = False  # default: reweighting mode

    def __call__(self, module: WindowAttention, inputs, outputs):
        x_out, attn = outputs
        B_, N, C    = x_out.shape
        head_dim    = C // self.num_heads

        # Ricalcola V dall'input (eval mode: nessuna discrepanza da attn_drop)
        qkv     = module.qkv(inputs[0]).reshape(B_, N, 3, self.num_heads, head_dim)
        v       = qkv.permute(2, 0, 3, 1, 4)[2]           # [B_, n_heads, N, head_dim]
        x_heads = attn @ v                                  # [B_, n_heads, N, head_dim]

        if self.collect_for_fitting:
            self._collect(x_heads, head_dim)
            return outputs

        # Reweighting mode: applica RD* per ogni head
        reweighted   = torch.stack(
            [self.spectral_layers[h](x_heads[:, h]) for h in range(self.num_heads)],
            dim=1,
        )                                                   # [B_, n_heads, N, head_dim]
        x_reweighted = reweighted.transpose(1, 2).reshape(B_, N, C)
        x_reweighted = module.proj(x_reweighted)
        x_reweighted = module.proj_drop(x_reweighted)

        return x_reweighted, attn

    def _collect(self, x_heads: torch.Tensor, head_dim: int):
        layer_key = f'layer_{self.layer_idx}'
        self.collected_data.setdefault(layer_key, {})
        self.collected_data[layer_key].setdefault(self.block_idx, {
            f'head_{h}': [] for h in range(self.num_heads)
        })
        for h in range(self.num_heads):
            self.collected_data[layer_key][self.block_idx][f'head_{h}'].append(
                x_heads[:, h].reshape(-1, head_dim).detach().cpu()
            )


# =============================================================================
# 3. RESIDUAL HTSAT
# =============================================================================

class ResiDualHTSAT(HTSAT_Swin_Transformer):
    """
    HTSAT con RD* spectral reweighting.

    Config minimale:
        residual_config = {
            'target_layers':      List[int],   # es. [1, 2, 3]
            'variance_threshold': float,       # 0.95 o 0.90
        }

    Pipeline:
        1. fit_pca_on_data(dataloader)          → calcola Φ, μ, inizializza λ
        2. optimize_lambda(val_loader, ...)     → ottimizza λ su zero-shot accuracy
        3. forward(audio)                       → inference con λ ottimizzati
    """

    def __init__(self, *args, residual_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_layers      = residual_config.get('target_layers', [2, 3])
        self.variance_threshold = residual_config.get('variance_threshold', 0.95)

        self.spectral_layers = nn.ModuleDict()
        self.collected_data: Dict = {}
        self._hooks: List[AttentionHook] = []  # riferimenti per set collect_for_fitting

        self._build_spectral_layers()
        self._register_hooks()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _build_spectral_layers(self):
        """Costruisce la ModuleDict degli SpectralReweightingLayer."""
        for layer_idx in self.target_layers:
            if layer_idx >= len(self.layers):
                print(f"[ResiDual] Layer {layer_idx} non esiste, skip")
                continue

            layer_dim  = int(self.embed_dim * (2 ** layer_idx))
            num_heads  = self.num_heads[layer_idx]
            head_dim   = layer_dim // num_heads
            num_blocks = self.depths[layer_idx]
            layer_name = f'layer_{layer_idx}'

            self.spectral_layers[layer_name] = nn.ModuleList([
                nn.ModuleList([
                    SpectralReweightingLayer(embed_dim=head_dim,
                                            variance_threshold=self.variance_threshold)
                    for _ in range(num_heads)
                ])
                for _ in range(num_blocks)
            ])

    def _register_hooks(self):
        """Registra gli AttentionHook in modo permanente su ogni WindowAttention target."""
        for layer_idx in self.target_layers:
            layer_name = f'layer_{layer_idx}'
            if layer_name not in self.spectral_layers:
                continue

            spectral_block = self.spectral_layers[layer_name]
            layer          = self.layers[layer_idx]

            for block_idx, block in enumerate(layer.blocks):
                hook = AttentionHook(
                    spectral_layers = spectral_block[block_idx],
                    num_heads       = block.attn.num_heads,
                    collected_data  = self.collected_data,
                    layer_idx       = layer_idx,
                    block_idx       = block_idx,
                )
                block.attn.register_forward_hook(hook)
                self._hooks.append(hook)

    def _set_collect_mode(self, collect: bool):
        for hook in self._hooks:
            hook.collect_for_fitting = collect

    # -------------------------------------------------------------------------
    # FORWARD!!
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False):
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

        # Modalità di inferenza con audio di lunghezza arbitraria (per il momento non ci interessa)
        if infer_mode:
            x = self._prepare_infer(x)
            return self._forward_features(x)

        if self.config.enable_repeat_mode:
            return self._forward_repeat_mode(x)

        return self._forward_standard_mode(x)

    def _prepare_infer(self, x):
        target_T     = int(self.spec_size * self.freq_ratio)
        repeat_ratio = math.floor(target_T / x.shape[2])
        return self.reshape_wav2img(x.repeat(1, 1, repeat_ratio, 1))

    def _forward_repeat_mode(self, x):
        if self.training:
            cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
            return self._forward_features(self.repeat_wat2img(x, cur_pos))
        outputs = [
            self._forward_features(self.repeat_wat2img(x.clone(), p))
            for p in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size)
        ]
        return self._average_outputs(outputs)

    def _forward_standard_mode(self, x):
        max_len = self.freq_ratio * self.spec_size
        if x.shape[2] <= max_len:
            return self._forward_features(self.reshape_wav2img(x))

        # Questa parte (sotto) non viene mai eseguita per via del crop deterministico a 7s
        if self.training:
            return self._forward_features(
                self.reshape_wav2img(self.crop_wav(x, crop_size=max_len))
            )
        overlap, crop = 344, 689
        outputs = [
            self._forward_features(self.reshape_wav2img(self.crop_wav(x, crop, p)))
            for p in range(0, x.shape[2] - crop - 1, overlap)
        ]
        return self._average_outputs(outputs)

    def _forward_features(self, x):
        frames_num = x.shape[2]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        attn = None
        for layer in self.layers:
            x, attn = layer(x)

        return self._finalize_output(x, attn, frames_num)

    # -------------------------------------------------------------------------
    # Finalize output (identico a CLAP originale)
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

        Args:
            dataloader:  DataLoader con audio
            max_samples: Numero massimo di campioni da raccogliere

        Returns:
            fit_info: Per ogni head → k, varianza spiegata, profilo eigenvalori
        """
        self.eval()
        self.collected_data.clear()
        self._set_collect_mode(True)

        n_samples = 0
        with torch.no_grad():
            for audio in tqdm(dataloader, desc="[ResiDual] Raccolta per PCA"):
                if n_samples >= max_samples:
                    break
                if next(self.parameters()).is_cuda and not audio.is_cuda:
                    audio = audio.cuda()
                try:
                    self.forward(audio)
                    n_samples += audio.size(0)
                except Exception as e:
                    print(f"[ResiDual] Errore: {e}")

        self._set_collect_mode(False)
        print(f"[ResiDual] Raccolti {n_samples} campioni")

        fit_info = {}
        for layer_name, block_data in self.collected_data.items():
            fit_info[layer_name] = {}
            block_spectral = self.spectral_layers[layer_name]

            for block_idx, head_data in block_data.items():
                fit_info[layer_name][f'block_{block_idx}'] = {}

                for head_name, data_list in head_data.items():
                    if not data_list:
                        continue
                    combined = torch.cat(data_list, dim=0)
                    head_idx = int(head_name.split('_')[1])
                    info     = block_spectral[block_idx][head_idx].fit_pca(combined)
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
                        max_epochs: int   = 30,
                        patience:   int   = 5,
                        lr:         float = 1e-2) -> Dict:
        """
        Ottimizza i pesi λ per massimizzare la zero-shot accuracy sul validation set.

        Solo i λ vengono aggiornati. Φ, μ e tutti gli altri parametri sono congelati.

        Args:
            val_dataloader:        DataLoader con (audio, label)
            class_text_embeddings: [n_classes, d_proj] — text embeddings pre-calcolati
            projection:            Projection layer (768 → 1024), congelata ma nel
                                   computational graph per far fluire i gradienti
            max_epochs:            Numero massimo di epoche
            patience:              Early stopping patience
            lr:                    Learning rate

        Returns:
            history: {'accuracy': [...], 'best_epoch': int, 'best_accuracy': float}
        """
        device = next(self.parameters()).device

        self._freeze_all_except_lambda()
        lambda_params = self._get_lambda_parameters()
        if not lambda_params:
            raise RuntimeError("[ResiDual] Nessun lambda trovato. Hai chiamato fit_pca_on_data?")

        optimizer = torch.optim.Adam(lambda_params, lr=lr)
        # Alternativa: schedulefree.AdamWScheduleFree(lambda_params, lr=lr)

        class_text_embeddings = nn.functional.normalize(
            class_text_embeddings.to(device), dim=-1
        )

        best_accuracy = -1.0
        best_lambda   = self._snapshot_lambda()
        epochs_no_imp = 0
        history       = {'accuracy': [], 'best_epoch': 0, 'best_accuracy': 0.0}

        for epoch in tqdm(range(max_epochs), desc="  Epochs", leave=True):
            acc = self._optimize_epoch(val_dataloader, class_text_embeddings,
                                       optimizer, device, projection)
            history['accuracy'].append(acc)
            print(f"  Epoch {epoch+1:3d}/{max_epochs} | acc={acc:.4f} | best={best_accuracy:.4f}")

            if acc > best_accuracy:
                best_accuracy            = acc
                best_lambda              = self._snapshot_lambda()
                epochs_no_imp            = 0
                history['best_epoch']    = epoch + 1
                history['best_accuracy'] = best_accuracy
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"[ResiDual] Early stopping a epoca {epoch+1}")
                    break

        self._restore_lambda(best_lambda)
        self._unfreeze_all()
        print(f"[ResiDual] Best accuracy: {best_accuracy:.4f} @ epoch {history['best_epoch']}")

        return history

    def _optimize_epoch(self, dataloader, class_text_embeddings, optimizer,
                        device, projection: nn.Module) -> float:
        """
        Un'epoca di ottimizzazione su zero-shot accuracy.

        Flusso del gradiente:
            cross_entropy → logits → audio_emb_norm → projection → latent_output
            → SpectralReweightingLayer → λ
        """
        self.eval()
        total_correct = 0
        total_samples = 0

        # Loop su tutti i batch del validation set
        for batch in tqdm(dataloader, desc="    Optimizing λ", leave=True):
            audio, labels = batch[0], batch[1]
            audio  = audio.to(device)
            labels = labels.to(device)

            if audio.dim() == 3:        # [B, 1, samples] → [B, samples]
                audio = audio.squeeze(1)
            
            # Reset dei gradienti
            optimizer.zero_grad()

            latent         = self.forward(audio)['latent_output']           # [B, 768]
            audio_emb_norm = nn.functional.normalize(projection(latent), dim=-1) # [B, 1024]

            # Calcolo similarità con tutte le classi (qui sono normalizzati quindi cosine similarity)
            logits         = audio_emb_norm @ class_text_embeddings.T       # [B, n_classes]
            
            # Loss cross-entropy + il gradiente fluisce fino ai λ
            nn.functional.cross_entropy(logits, labels).backward()

            # Aggiornamento dei parametri λ
            optimizer.step()

            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.size(0)

        return total_correct / total_samples if total_samples > 0 else 0.0

    # -------------------------------------------------------------------------
    # Utilities per gestione λ
    # -------------------------------------------------------------------------

    # Ottengo tutti i parametri λ che devono essere ottimizzati da tutti i layer fitted
    def _get_lambda_parameters(self) -> List[nn.Parameter]:
        return [
            layer.lambda_weights
            for layer_modules in self.spectral_layers.values()
            for block_modules in layer_modules
            for layer in block_modules
            if layer.is_fitted
        ]
    
    # Congela tutti i parametri tranne quelli λ
    def _freeze_all_except_lambda(self):
        lambda_ids = {id(p) for p in self._get_lambda_parameters()}
        for p in self.parameters():
            if id(p) not in lambda_ids:
                p.requires_grad_(False)
    
    # Scongela tutti i parametri del modello
    def _unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    # Salvo tutti i parametri λ per ogni testa
    def _snapshot_lambda(self) -> Dict[str, torch.Tensor]:
        return {
            f'{ln}_{bi}_{hi}': layer.lambda_weights.detach().clone()
            for ln, layer_modules in self.spectral_layers.items()
            for bi, block_modules in enumerate(layer_modules)
            for hi, layer in enumerate(block_modules)
        }

    # Setto tutti i parametri λ relativi allo snapshot migliore
    def _restore_lambda(self, snapshot: Dict[str, torch.Tensor]):
        for ln, layer_modules in self.spectral_layers.items():
            for bi, block_modules in enumerate(layer_modules):
                for hi, layer in enumerate(block_modules):
                    key = f'{ln}_{bi}_{hi}'
                    if key in snapshot:
                        with torch.no_grad():
                            # Con copy_ modifica in-place
                            layer.lambda_weights.copy_(snapshot[key])

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

        model = ResiDualCLAP(..., residual_config={
            'target_layers':      [1, 2, 3],
            'variance_threshold': 0.95,
        })
        model.load_state_dict(clap_weights)

        fit_info = model.fit_spectral_components(train_loader)
        text_embs = model.encode_text(class_prompts)       # [n_classes, d_proj]
        history = model.optimize_spectral_weights(val_loader, text_embs)

        audio_emb, _ = model.audio_encoder(audio)          # inference

    Config per dataset generico (ESC-50):   target_layers=[1,2,3], variance_threshold=0.95
    Config per dataset specifico (VocalSound): target_layers=[2,3], variance_threshold=0.90
    """

    def __init__(self, *args, residual_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_encoder = AudioEncoder(
            kwargs["audioenc_name"], kwargs["out_emb"],     kwargs["d_proj"],
            kwargs["sample_rate"],   kwargs["window_size"], kwargs["hop_size"],
            kwargs["mel_bins"],      kwargs["fmin"],        kwargs["fmax"],
            kwargs["classes_num"],   residual_config,
        )

    def fit_spectral_components(self, audio_dataloader, max_samples: int = 10000) -> Dict:
        """Fase 1: calcola Φ, μ e inizializza λ per ogni head target."""
        return self.audio_encoder.base.htsat.fit_pca_on_data(audio_dataloader, max_samples)

    def optimize_spectral_weights(self,
                                  val_dataloader,
                                  class_text_embeddings: torch.Tensor,
                                  max_epochs: int   = 30,
                                  patience:   int   = 5,
                                  lr:         float = 1e-2) -> Dict:
        """
        Fase 2: ottimizza λ su zero-shot accuracy.

        La projection (768 → 1024) è congelata ma inclusa nel computational graph
        affinché i gradienti fluiscano fino ai λ.
        """
        return self.audio_encoder.base.htsat.optimize_lambda(
            val_dataloader,
            class_text_embeddings,
            projection = self.audio_encoder.projection,
            max_epochs = max_epochs,
            patience   = patience,
            lr         = lr,
        )