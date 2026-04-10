"""
ResiDual HTSAT: Spectral Reweighting per Audio Transformer

Due modalità operative:
1. ATTENTION MODE: Reweighting per-head (ogni testa di attenzione ha il suo PCA)
2. LAYER MODE: Reweighting per-layer (un PCA per l'output completo del layer)

═══════════════════════════════════════════════════════════════════════════════
LEGENDA DIMENSIONI TENSOR
═══════════════════════════════════════════════════════════════════════════════

Dimensioni batch e finestre:
  B        : Batch size (numero di campioni audio nel batch)
  B_       : Batch effettivo dopo window partition = B * nW
               (ogni sample è diviso in nW finestre non sovrapposte)
  nW       : Numero di finestre per sample = (H / Wh) * (W / Ww)

Dimensioni spaziali del feature map:
  H        : Altezza del feature map dopo patch embedding (asse frequenza)
  W        : Larghezza del feature map dopo patch embedding (asse tempo)
  Wh       : Altezza della finestra di attenzione (window_size[0])
  Ww       : Larghezza della finestra di attenzione (window_size[1])
  N        : Numero di token per finestra = Wh * Ww
  SF       : Numero di frame spaziali (frequenza) dopo tutti i layer
  ST       : Numero di frame spaziali (tempo) dopo tutti i layer
  F        : Dimensione frequenza del feature map 4D
  T        : Dimensione tempo del feature map 4D

Dimensioni dei canali / embedding:
  C        : Dimensione embedding nel layer corrente
               (raddoppia ad ogni layer: embed_dim * 2^layer_idx)
  embed_dim: Dimensione embedding base (es. 96)
  d_in     : Dimensione input al Projection layer
  d_out    : Dimensione output del Projection layer (spazio multimodale)

Dimensioni specifiche dell'attenzione:
  num_heads: Numero di teste di attenzione nel layer corrente
  head_dim : Dimensione per singola testa = C // num_heads
  head_idx : Indice della testa corrente (0 .. num_heads-1)

Dimensioni PCA / Spectral Reweighting:
  n_comp   : Numero di componenti PCA (n_components)
  N_pca    : Numero di campioni raccolti per il fit PCA
               (= B_ * N per attention mode, = B * H * W per layer mode)

Dimensioni specifiche audio / spettrali:
  freq_bins: Numero di bin frequenziali del mel-spettrogramma
  time_frames: Numero di frame temporali dello spettrogramma
  c_freq_bin: Numero di bin di frequenza per bin TSCAM

Classi:
  classes_num: Numero di classi di classificazione audio
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import numpy as np
import random
import math
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Dict
from .clap import CLAP, Projection
from .htsat import HTSAT_Swin_Transformer, WindowAttention


# ============================================================================
# SPECTRAL REWEIGHTING LAYER - Core del reweighting PCA
# ============================================================================

class SpectralReweightingLayer(nn.Module):
    """
    Layer che applica whitening spettrale tramite PCA.
    
    Pipeline:
    1. fit_pca(data): Calcola PCA su dati raccolti
       - Trova componenti principali che catturano la varianza
       - Calcola pesi di whitening: w_j = λ_j^{-γ/2}
    
    2. forward(x): Applica whitening
       - Centra i dati (x - mean)
       - Proietta sulle componenti principali
       - Applica whitening: scala ogni PC per w_j
       - Ricostruisce + aggiunge residuo + de-centra
    """
    
    def __init__(self, embed_dim: int,
                 n_components: int = None,
                 whitening_strength: float = 1.0,
                 whitening_eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_components = n_components or embed_dim // 4
        self.whitening_strength = whitening_strength  # γ: 0=identità, 1=whitening completo
        self.whitening_eps = whitening_eps             # clamp per stabilità numerica

        # Buffer shapes (statici, inizializzati prima del fit):
        self.register_buffer('pc_weights',     torch.ones(self.n_components))          # (n_comp,)
        self.register_buffer('pca_components', torch.eye(embed_dim, self.n_components))# (embed_dim, n_comp)
        self.register_buffer('pca_mean',       torch.zeros(embed_dim))                 # (embed_dim,)
        self.register_buffer('is_fitted',      torch.tensor(False))                    # scalar bool

    def fit_pca(self, data: torch.Tensor):
        """
        Calcola PCA e inizializza i pesi di whitening.

        Args:
            data: Tensor (N_pca, embed_dim)
                  In attention mode: N_pca = somma sui batch di (B_ * N)
                  In layer mode:     N_pca = somma sui batch di (B * H * W)

        Returns:
            variance_ratio: Array (n_comp,) con % di varianza spiegata da ogni PC
        """
        # data: (N_pca, embed_dim)
        X = data.detach().cpu().numpy()
        # X: (N_pca, embed_dim)  [numpy]

        pca = PCA(n_components=self.n_components)
        pca.fit(X)

        # pca.components_: (n_comp, embed_dim)  → trasposto → (embed_dim, n_comp)
        self.pca_components.data = torch.tensor(pca.components_.T, dtype=torch.float32)
        # pca_components: (embed_dim, n_comp)

        # pca.mean_: (embed_dim,)
        self.pca_mean.data = torch.tensor(pca.mean_, dtype=torch.float32)
        # pca_mean: (embed_dim,)

        # eigenvalues: (n_comp,)
        eigenvalues = pca.explained_variance_
        # w_j = λ_j^{-γ/2}, clampato per stabilità numerica
        weights = 1.0 / np.maximum(
            eigenvalues ** (self.whitening_strength / 2.0),
            self.whitening_eps
        )
        # weights: (n_comp,)
        self.pc_weights.data = torch.tensor(weights, dtype=torch.float32)
        # pc_weights: (n_comp,)

        self.is_fitted.data = torch.tensor(True)
        # pca.explained_variance_ratio_: (n_comp,)
        return pca.explained_variance_ratio_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applica whitening spettrale.

        Args:
            x: Input tensor (..., embed_dim)
               In attention mode: (B_, N, head_dim)   con embed_dim = head_dim
               In layer mode:     (B, H*W, C)         con embed_dim = C

        Returns:
            Tensor whitened con stessa shape di x
        """
        if not self.is_fitted:
            return x

        original_shape = x.shape
        # original_shape: (..., embed_dim)

        x_flat = x.reshape(-1, self.embed_dim)
        # x_flat: (N_flat, embed_dim)
        #   dove N_flat = prod(original_shape[:-1])
        #   es. attention mode: N_flat = B_ * N
        #   es. layer mode:     N_flat = B * H * W

        x_centered = x_flat - self.pca_mean
        # x_flat:    (N_flat, embed_dim)
        # pca_mean:  (embed_dim,)          [broadcast]
        # x_centered:(N_flat, embed_dim)

        pc_proj = torch.matmul(x_centered, self.pca_components)
        # x_centered:     (N_flat, embed_dim)
        # pca_components: (embed_dim, n_comp)
        # pc_proj:        (N_flat, n_comp)

        weighted_proj = pc_proj * self.pc_weights
        # pc_proj:       (N_flat, n_comp)
        # pc_weights:    (n_comp,)          [broadcast]
        # weighted_proj: (N_flat, n_comp)

        reconstructed = torch.matmul(weighted_proj, self.pca_components.T)
        # weighted_proj:    (N_flat, n_comp)
        # pca_components.T: (n_comp, embed_dim)
        # reconstructed:    (N_flat, embed_dim)

        residual = x_centered - torch.matmul(pc_proj, self.pca_components.T)
        # pc_proj:          (N_flat, n_comp)
        # pca_components.T: (n_comp, embed_dim)
        # matmul result:    (N_flat, embed_dim)   [proiezione ricostruita senza scaling]
        # x_centered:       (N_flat, embed_dim)
        # residual:         (N_flat, embed_dim)   [componenti fuori dal sottospazio PCA]

        #return (reconstructed  + self.pca_mean).reshape(original_shape)
        return (reconstructed + residual + self.pca_mean).reshape(original_shape) 
        # (N_flat, embed_dim) → reshape → original_shape (es. B_, N, head_dim)

# ============================================================================
# ATTENTION HEAD REWEIGHTING - Per modalità 'attention'
# ============================================================================

class WindowAttentionReweighting(WindowAttention):
    """
    WindowAttention modificata per applicare reweighting PER TESTA.
    
    Pipeline:
    1. Calcola Q, K, V come standard
    2. Calcola attention weights
    3. Output per testa: head_i = attention_i @ V_i
    4. RACCOLTA (opzionale): salva head_i in collected_data
    5. REWEIGHTING: head_i_reweighted = spectral_layer_i(head_i)
    6. Concatena teste reweighted
    7. Projection finale
    """
    
    def __init__(self, *args, spectral_layers: nn.ModuleList = None, 
                 layer_idx: int = None, block_idx: int = None, 
                 collected_data: dict = None, **kwargs):
        """
        Args:
            spectral_layers: ModuleList con uno SpectralLayer per ogni testa
            layer_idx: Indice del layer (per identificazione durante raccolta)
            block_idx: Indice del blocco dentro il layer (per identificazione)
            collected_data: Dizionario condiviso per raccolta dati (riferimento)
        """
        super().__init__(*args, **kwargs)
        self.spectral_layers = spectral_layers
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.collected_data = collected_data  # Riferimento al dict del modello padre
    
    def forward(self, x, mask=None, collect_for_fitting=False):
        """
        Forward con reweighting per-head e raccolta opzionale.
        
        Args:
            x: Input (B_, N, C)
               B_ = B * nW  (batch size × numero di finestre per sample)
               N  = Wh * Ww (token per finestra)
               C  = embed_dim del layer corrente
            mask: Attention mask opzionale (nW, N, N)
            collect_for_fitting: Se True, salva head outputs in collected_data
            
        Returns:
            x: Output reweighted (B_, N, C)
            attn: Attention weights (B_, num_heads, N, N)
        """
        B_, N, C = x.shape
        # x: (B_, N, C)

        # Step 1: Calcola Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        # self.qkv(x): (B_, N, 3*C)
        # dopo reshape:  (B_, N, 3, num_heads, head_dim)   con head_dim = C // num_heads
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv: (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: (B_, num_heads, N, head_dim)
        
        # Step 2: Calcola attention scores
        q = q * self.scale
        # q: (B_, num_heads, N, head_dim)   [scalato per 1/sqrt(head_dim)]
        attn = (q @ k.transpose(-2, -1))
        # q:              (B_, num_heads, N, head_dim)
        # k.transpose:    (B_, num_heads, head_dim, N)
        # attn:           (B_, num_heads, N, N)
        
        # Step 3: Aggiungi relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1) # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # relative_position_bias: (num_heads, N, N)   con N = Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        # relative_position_bias.unsqueeze(0): (1, num_heads, N, N)  [broadcast su B_]
        # attn: (B_, num_heads, N, N)
        
        # Step 4: Applica mask se presente
        if mask is not None:
            nW = mask.shape[0]
            # mask: (nW, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            # attn: (B, nW, num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            # mask.unsqueeze(1).unsqueeze(0): (1, nW, 1, N, N) [broadcast]
            # attn: (B, nW, num_heads, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            # attn: (B_, num_heads, N, N)   con B_ = B * nW
        
        attn = self.softmax(attn)
        # attn: (B_, num_heads, N, N)   [somma a 1 lungo ultima dim]
        attn = self.attn_drop(attn)
        # attn: (B_, num_heads, N, N)
        
        # Step 5: Calcola output attention per ogni testa
        x_heads = attn @ v
        # attn:    (B_, num_heads, N, N)
        # v:       (B_, num_heads, N, head_dim)
        # x_heads: (B_, num_heads, N, head_dim)

        # Step 6: RACCOLTA DATI (se richiesto)
        if collect_for_fitting and self.collected_data is not None:
            self._collect_heads(x_heads)
        
        # Step 7: REWEIGHTING PER-HEAD (se fitted)
        if self.spectral_layers is not None:
            reweighted_heads = []
            for head_idx in range(self.num_heads):
                head_output = x_heads[:, head_idx, :, :]
                # head_output: (B_, N, head_dim)   [testa head_idx estratta]
                reweighted = self.spectral_layers[head_idx](head_output)
                # reweighted:  (B_, N, head_dim)   [dopo SpectralReweightingLayer]
                reweighted_heads.append(reweighted)
            x_heads = torch.stack(reweighted_heads, dim=1)
            # reweighted_heads: lista di num_heads tensori (B_, N, head_dim)
            # x_heads:          (B_, num_heads, N, head_dim)

        # Step 8: Concatena teste e applica projection finale
        x = x_heads.transpose(1, 2).reshape(B_, N, C)
        # x_heads.transpose(1,2): (B_, N, num_heads, head_dim)
        # dopo reshape:            (B_, N, C)   con C = num_heads * head_dim
        x = self.proj(x)
        # x: (B_, N, C)   [linear projection]
        x = self.proj_drop(x)
        # x: (B_, N, C)
        
        return x, attn
    
    def _collect_heads(self, x_heads):
        """
        Salva output delle teste nel dizionario condiviso.
        
        Args:
            x_heads: Tensor (B_, num_heads, N, head_dim)
        """
        layer_name = f'layer_{self.layer_idx}'
        
        # Inizializza struttura del layer se necessario
        if layer_name not in self.collected_data:
            self.collected_data[layer_name] = {}
        
        # Inizializza struttura per blocco se necessario
        if self.block_idx not in self.collected_data[layer_name]:
            self.collected_data[layer_name][self.block_idx] = {
                f'head_{i}': [] for i in range(self.num_heads)
            }
        
        # Salva ogni testa
        for head_idx in range(self.num_heads):
            head_data = x_heads[:, head_idx, :, :]
            # x_heads:   (B_, num_heads, N, head_dim)
            # head_data: (B_, N, head_dim)   [testa head_idx]
            reshaped = head_data.reshape(-1, head_data.size(-1))
            # head_data: (B_, N, head_dim)
            # reshaped:  (B_ * N, head_dim)   [appiattito per PCA]
            self.collected_data[layer_name][self.block_idx][f'head_{head_idx}'].append(
                reshaped.detach().cpu()
            )
            # collected: lista di tensori (B_ * N, head_dim) per ogni batch

# ============================================================================
# RESIDUAL HTSAT - Modello principale
# ============================================================================

class ResiDualHTSAT(HTSAT_Swin_Transformer):
    """
    HTSAT con spectral reweighting configurabile.
    
    MODALITÀ 'attention':
    - Crea uno SpectralLayer PER OGNI TESTA nei layer target
    - Reweighting applicato sull'output di ogni testa separatamente
    - Durante fitting: raccoglie output di ogni testa (head_dim dimensioni)
    - Durante inference: WindowAttentionReweighting applica reweighting per-head
    
    MODALITÀ 'layer':
    - Crea uno SpectralLayer PER LAYER target
    - Reweighting applicato sull'output completo del layer
    - Durante fitting: raccoglie output completo (layer_dim dimensioni)
    - Durante inference: applica reweighting dopo forward del layer
    
    Args:
        residual_config: Configurazione per spectral reweighting
    """
    
    def __init__(self, *args, residual_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if residual_config is None:
            raise ValueError("residual_config è obbligatorio")
        
        self.residual_config = residual_config
        self.mode = residual_config.get('mode', 'layer')
        self.target_layers = residual_config.get('target_layers', [])
        
        # Storage per spectral layers
        # - Mode 'attention': dict[layer_idx] -> ModuleList[SpectralLayer per testa]
        # - Mode 'layer': dict[layer_idx] -> SpectralLayer
        self.spectral_layers = nn.ModuleDict()
        
        # Buffer temporaneo per raccolta dati durante fitting
        self.collected_data = {}
        
        # Setup: crea spectral layers e inietta in attention se necessario
        self._setup_spectral_layers()
        
    def _setup_spectral_layers(self):
        """
        Crea gli spectral layers in base alla modalità configurata.
        
        ATTENTION MODE:
        - Per ogni layer target, crea num_blocks × num_heads SpectralLayers
        - Ogni blocco ha le sue teste, quindi layer_0 con 2 blocchi e 4 heads = 8 SpectralLayers
        - Identificazione: layer_X_block_Y_head_Z
        
        LAYER MODE:
        - Per ogni layer target, crea un singolo SpectralLayer
        - Dimensione = layer_dim completo
        """
        n_components_ratio = self.residual_config.get('n_components_ratio', 0.25)
        
        print(f"\n{'='*80}")
        print(f"🔧 Setup ResiDual HTSAT")
        print(f"{'='*80}")
        print(f"Modalità: {self.mode.upper()}")
        print(f"Target layers: {self.target_layers}")
        print(f"PCA components ratio: {n_components_ratio}")
        
        for layer_idx in self.target_layers:
            if layer_idx >= len(self.layers):
                print(f"⚠️  Layer {layer_idx} non esiste (max: {len(self.layers)-1}), skip")
                continue
            
            # Dimensione embedding del layer: raddoppia ad ogni stage Swin
            # layer_dim = embed_dim * 2^layer_idx
            # es. embed_dim=96: layer_0→96, layer_1→192, layer_2→384, layer_3→768
            layer_dim = int(self.embed_dim * (2 ** layer_idx))
            layer_name = f'layer_{layer_idx}'
            
            if self.mode == 'attention':
                # MODE ATTENTION: Crea uno spectral layer per ogni testa di ogni blocco
                num_heads = self.num_heads[layer_idx]   # numero teste in questo layer
                num_blocks = self.depths[layer_idx]     # numero blocchi in questo layer
                head_dim = layer_dim // num_heads       # dim per testa = C // num_heads
                n_components = int(head_dim * n_components_ratio)
                # SpectralReweightingLayer lavora su spazio (head_dim,) → (n_components,) PCs

                # Crea ModuleList di ModuleList: [blocco][testa]
                # Ogni blocco ha le sue teste, ognuna con il proprio SpectralLayer
                # spectral input/output shape per ogni layer: (B_, N, head_dim)
                block_spectral_layers = nn.ModuleList([
                    nn.ModuleList([
                        SpectralReweightingLayer(
                            embed_dim=head_dim,      # dim dello spazio su cui opera la PCA
                            n_components=n_components,
                        )
                        for _ in range(num_heads)    # uno per testa
                    ])
                    for _ in range(num_blocks)       # uno per blocco
                ])
                
                self.spectral_layers[layer_name] = block_spectral_layers
                
                total_heads = num_blocks * num_heads
                print(f"\n✓ {layer_name}:")
                print(f"  Modalità: PER-HEAD reweighting")
                print(f"  Num blocks: {num_blocks}")
                print(f"  Heads per block: {num_heads}")
                print(f"  Total heads: {total_heads}")
                print(f"  Head dim: {head_dim}D → {n_components} PCs")
                
                # Inietta reweighting nelle WindowAttention
                self._inject_attention_reweighting(layer_idx, block_spectral_layers)
                
            else:  # mode == 'layer'
                # MODE LAYER: Crea un singolo spectral layer per l'intero output
                # SpectralReweightingLayer opera su (layer_dim,) dopo downsampling
                n_components = int(layer_dim * n_components_ratio)
                # spectral input/output shape: (B, H_out*W_out, layer_dim)
                #   dove H_out, W_out sono le dimensioni dopo il downsampling del layer

                self.spectral_layers[layer_name] = SpectralReweightingLayer(
                    embed_dim=layer_dim,       # dim dello spazio su cui opera la PCA
                    n_components=n_components,
                )
                
                print(f"\n✓ {layer_name}:")
                print(f"  Modalità: LAYER reweighting")
                print(f"  Layer dim: {layer_dim}D → {n_components} PCs")
        
        print(f"\n{'='*80}\n")
    
    def _inject_attention_reweighting(self, layer_idx: int, block_spectral_layers: nn.ModuleList):
        """
        Sostituisce WindowAttention con WindowAttentionReweighting nei blocchi target.
        
        Passa layer_idx, block_idx e riferimento a collected_data per permettere
        la raccolta automatica durante forward.
        
        Args:
            layer_idx: Indice del layer dove iniettare
            block_spectral_layers: ModuleList[ModuleList[SpectralLayer]]
                shape logica: [num_blocks][num_heads] → SpectralReweightingLayer(head_dim)
        """
        target_layer = self.layers[layer_idx]
        
        for block_idx, block in enumerate(target_layer.blocks):
            old_attn = block.attn
            spectral_layers_for_block = block_spectral_layers[block_idx]
            
            # Estratti dropout rates in modo sicuro
            attn_drop_rate = old_attn.attn_drop.p if hasattr(old_attn.attn_drop, 'p') else 0.
            proj_drop_rate = old_attn.proj_drop.p if hasattr(old_attn.proj_drop, 'p') else 0.
            
            # Crea nuova attention con metadati per raccolta
            block.attn = WindowAttentionReweighting(
                dim=old_attn.dim,
                # dim: C = embed_dim * 2^layer_idx
                window_size=old_attn.window_size,
                # window_size: (Wh, Ww)
                num_heads=old_attn.num_heads,
                qkv_bias=True,
                qk_scale=old_attn.scale,  # Verrà calcolato automaticamente come head_dim ** -0.5
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                spectral_layers=spectral_layers_for_block,
                # spectral_layers: ModuleList di num_heads SpectralReweightingLayer(head_dim)
                layer_idx=layer_idx,
                block_idx=block_idx,
                collected_data=self.collected_data  # Riferimento condiviso
            )
            
            # Copia pesi dal vecchio modulo
            block.attn.load_state_dict(old_attn.state_dict(), strict=False)
        
    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False, collect_for_fitting=False):
        """
        Forward pass con opzione di raccolta dati per fitting PCA.
        
        Args:
            x: Input audio waveform (B, audio_samples)
            mixup_lambda: Coefficiente per mixup durante training
            infer_mode: Se True, gestisce input di lunghezza variabile
            collect_for_fitting: Se True, salva hidden states per fitting PCA
        
        Returns:
            output_dict: Dizionario con clipwise_output, framewise_output, etc.
        """
        
        # ========== PREPROCESSING AUDIO ==========
        # Identico a HTSAT originale: spectrogram -> logmel -> normalization
        x = self.spectrogram_extractor(x)
        # x: (B, 1, time_frames, freq_bins)   [spettrogramma raw]
        x = self.logmel_extractor(x)
        # x: (B, 1, time_frames, mel_bins)    [scala log-mel]
        x = x.transpose(1, 3)
        # x: (B, mel_bins, time_frames, 1)
        x = self.bn0(x)
        # x: (B, mel_bins, time_frames, 1)    [batch norm]
        x = x.transpose(1, 3)
        # x: (B, 1, time_frames, mel_bins)    [torna al formato originale]

        # Augmentation durante training
        if self.training:
            x = self.spec_augmenter(x)
            # x: (B, 1, time_frames, mel_bins)  [augmented]
        if self.training and mixup_lambda is not None:
            from .pytorch_utils import do_mixup
            x = do_mixup(x, mixup_lambda)
            # x: (B, 1, time_frames, mel_bins)  [mixup applicato]
        
        # ========== GESTIONE VARI INPUT MODE ==========
        # Logica identica a HTSAT originale per compatibilità
        
        if infer_mode:
            # Inference mode: ripeti spectrogram per raggiungere target size
            frame_num = x.shape[2]
            # frame_num: numero di frame temporali dello spettrogramma
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
            # x: (B, 1, time_frames * repeat_ratio, mel_bins)
            x = self.reshape_wav2img(x)
            # x: (B, 1, spec_size, spec_size)   [riformattato come immagine quadrata]
            output_dict = self._forward_features(x, collect_for_fitting)
        
        elif self.config.enable_repeat_mode:
            # Repeat mode: processa multiple posizioni e media
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                # x: (B, 1, spec_size, spec_size)
                output_dict = self._forward_features(x, collect_for_fitting)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    # tx: (B, 1, time_frames, mel_bins)
                    tx = self.repeat_wat2img(tx, cur_pos)
                    # tx: (B, 1, spec_size, spec_size)
                    output_dicts.append(self._forward_features(tx, False))
                output_dict = self._average_outputs(output_dicts, x.device)
        
        else:
            # Standard mode
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    # Training: random crop
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    # x: (B, 1, freq_ratio * spec_size, mel_bins)
                    x = self.reshape_wav2img(x)
                    # x: (B, 1, spec_size, spec_size)
                    output_dict = self._forward_features(x, collect_for_fitting)
                else:
                    # Inference: overlapping crops + media
                    overlap_size = 344
                    crop_size = 689
                    output_dicts = []
                    
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
                        # tx: (B, 1, crop_size, mel_bins)
                        tx = self.reshape_wav2img(tx)
                        # tx: (B, 1, spec_size, spec_size)
                        output_dicts.append(self._forward_features(tx, False))
                    
                    output_dict = self._average_outputs(output_dicts, x.device, include_latent=True)
            else:
                # Fase di interesse per questa versione ResidualCLAP
                x = self.reshape_wav2img(x)
                # x: (B, 1, spec_size, spec_size)
                output_dict = self._forward_features(x, collect_for_fitting)
        
        return output_dict
    
    def _forward_features(self, x, collect_for_fitting=False):
        """
        Core transformer con gestione raccolta/applicazione reweighting.
        
        MODE ATTENTION:
        - collect_for_fitting=True: WindowAttentionReweighting salva automaticamente
        - collect_for_fitting=False: WindowAttentionReweighting applica reweighting
        
        MODE LAYER:
        - collect_for_fitting=True: salva output layer in self.collected_data
        - collect_for_fitting=False: applica reweighting dopo layer forward
        
        Args:
            x: Input dopo preprocessing (B, 1, spec_size, spec_size)
            collect_for_fitting: Flag per raccolta dati PCA
            
        Returns:
            output_dict: Clipwise/framewise outputs
        """
        frames_num = x.shape[2]
        # frames_num: spec_size (numero di frame dopo reshape, usato per ricostruire le dim spaziali)

        # ========== PATCH EMBEDDING ==========
        x = self.patch_embed(x)
        # x: (B, H*W, embed_dim)
        #   H = spec_size // patch_size[0]  (numero patch sull'asse freq)
        #   W = spec_size // patch_size[1]  (numero patch sull'asse tempo)
        #   embed_dim: dimensione embedding base (es. 96)
        if self.ape:
            x = x + self.absolute_pos_embed
            # absolute_pos_embed: (1, H*W, embed_dim)  [broadcast su B]
            # x: (B, H*W, embed_dim)
        x = self.pos_drop(x)
        # x: (B, H*W, embed_dim)

        # ========== TRANSFORMER LAYERS ==========
        for i, layer in enumerate(self.layers):
            layer_name = f'layer_{i}'
            # Ogni layer Swin riduce la risoluzione spaziale di 2x (downsampling)
            # e raddoppia la dimensione dei canali: C_i = embed_dim * 2^i
            
            if self.mode == 'attention' and layer_name in self.spectral_layers and collect_for_fitting:
                # MODE ATTENTION con FITTING: Forward manuale dei blocchi con flag di raccolta
                # x all'ingresso del layer: (B, H_i * W_i, C_i)
                #   H_i = H // 2^i,  W_i = W // 2^i,  C_i = embed_dim * 2^i
                attns = []
                for block_idx, block in enumerate(layer.blocks):
                    if layer.use_checkpoint:
                        from torch.utils import checkpoint
                        x = checkpoint.checkpoint(block, x)
                    else:
                        # Forward completo del blocco (non solo attention)
                        shortcut = x
                        # shortcut: (B, H_i * W_i, C_i)   [per connessione residua]
                        x = block.norm1(x)
                        # x: (B, H_i * W_i, C_i)

                        x_shape = x.shape
                        # x_shape: (B, H_i * W_i, C_i)

                        # Reshape to 4D
                        x = x.reshape(x_shape[0], block.input_resolution[0], block.input_resolution[1], x_shape[2])
                        # x: (B, H_i, W_i, C_i)
                        
                        # Cyclic shift
                        if block.shift_size > 0:
                            shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                        else:
                            shifted_x = x
                        # shifted_x: (B, H_i, W_i, C_i)
                        
                        # Window partition
                        from .htsat import window_partition, window_reverse
                        x_windows = window_partition(shifted_x, block.window_size)
                        # shifted_x:  (B, H_i, W_i, C_i)
                        # x_windows:  (B * nW, Wh, Ww, C_i)
                        #   nW = (H_i // Wh) * (W_i // Ww)
                        x_windows = x_windows.reshape(-1, block.window_size * block.window_size, x_shape[2])
                        # x_windows: (B_ , N, C_i)   con B_ = B*nW, N = Wh*Ww
                        
                        # Attention con flag di raccolta
                        attn_windows, attn = block.attn(x_windows, mask=block.attn_mask, collect_for_fitting=True)
                        # attn_windows: (B_, N, C_i)   [output attention reweighted]
                        # attn:         (B_, num_heads, N, N)   [pesi di attenzione]
                        
                        # Window reverse
                        attn_windows = attn_windows.reshape(-1, block.window_size, block.window_size, x_shape[2])
                        # attn_windows: (B_, Wh, Ww, C_i)
                        shifted_x = window_reverse(attn_windows, block.window_size, 
                                                block.input_resolution[0], block.input_resolution[1])
                        # shifted_x: (B, H_i, W_i, C_i)
                        
                        # Reverse cyclic shift
                        if block.shift_size > 0:
                            x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                        else:
                            x = shifted_x
                        # x: (B, H_i, W_i, C_i)
                        
                        x = x.reshape(x_shape[0], block.input_resolution[0] * block.input_resolution[1], x_shape[2])
                        # x: (B, H_i * W_i, C_i)
                        
                        # FFN
                        x = shortcut + block.drop_path(x)
                        # x: (B, H_i * W_i, C_i)   [connessione residua post-attention]
                        x = x + block.drop_path(block.mlp(block.norm2(x)))
                        # x: (B, H_i * W_i, C_i)   [connessione residua post-MLP]
                        
                        if not self.training:
                            # Salva attn per il layer corrente (solo per analisi, non per il reweighting)
                            attns.append(attn.unsqueeze(0))
                            # attn.unsqueeze(0): (1, B_, num_heads, N, N)
                
                # Downsampling
                if layer.downsample is not None:
                    x = layer.downsample(x)
                    # x prima:  (B, H_i * W_i,     C_i)
                    # x dopo:   (B, H_i/2 * W_i/2, C_i * 2)   [patch merging]
                
                # Prepara attn
                if not self.training and attns:
                    attn = torch.cat(attns, dim=0)
                    # attn: (num_blocks, B_, num_heads, N, N)
                    attn = torch.mean(attn, dim=0)
                    # attn: (B_, num_heads, N, N)   [media sui blocchi]
                else:
                    attn = None
            
            elif self.mode == 'layer' and layer_name in self.spectral_layers:
                # MODE LAYER: Forward manuale per intercettare dopo il downsampling
                # x all'ingresso: (B, H_i * W_i, C_i)

                # 1. Forward attraverso i blocchi del layer
                attns = []
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        from torch.utils import checkpoint
                        x = checkpoint.checkpoint(blk, x)
                    else:
                        x, attn = blk(x)
                        # x:    (B, H_i * W_i, C_i)   [output del blocco Swin]
                        # attn: (B_, num_heads, N, N)  [pesi di attenzione]
                        if not self.training:
                            attns.append(attn.unsqueeze(0))
                            # attn.unsqueeze(0): (1, B_, num_heads, N, N)

                # 2. Applica downsampling PRIMA del reweighting  ← SPOSTATO QUI
                if layer.downsample is not None:
                    x = layer.downsample(x)
                    # x prima:  (B, H_i * W_i,     C_i)
                    # x dopo:   (B, H_i/2 * W_i/2, C_i * 2)   [patch merging]

                # 3. Raccogli/applica reweighting DOPO il downsampling  ← ORA QUI
                if collect_for_fitting:
                    # FITTING: Salva output DOPO il downsampling
                    # x: (B, H_out * W_out, C_out)
                    #   C_out = C_i * 2 (dopo patch merging)
                    if layer_name not in self.collected_data:
                        self.collected_data[layer_name] = []
                    self.collected_data[layer_name].append(
                        x.reshape(-1, x.size(-1)).detach().cpu()
                    )
                    # x.reshape(-1, x.size(-1)): (B * H_out * W_out, C_out)
                    # salvato come (N_pca_batch, C_out) da concatenare in seguito
                else:
                    # INFERENCE: Applica reweighting DOPO il downsampling
                    x = self.spectral_layers[layer_name](x)
                    # x: (B, H_out * W_out, C_out)   [dopo SpectralReweightingLayer]

                # 4. Prepara attn per output
                if not self.training and attns:
                    attn = torch.cat(attns, dim=0)
                    # attn: (num_blocks, B_, num_heads, N, N)
                    attn = torch.mean(attn, dim=0)
                    # attn: (B_, num_heads, N, N)
                else:
                    attn = None
            
            else:
                # Forward normale del layer (senza reweighting o con reweighting già integrato)
                x, attn = layer(x)
                # x:    (B, H_out * W_out, C_out)   [output del layer con eventuale downsampling]
                # attn: (B_, num_heads, N, N)
        
        # x al termine dei layer: (B, H_final * W_final, C_final)
        #   C_final = embed_dim * 2^(num_layers-1)   [es. 768 per 4 layer con embed_dim=96]

        # ========== FINALIZE OUTPUT ==========
        return self._finalize_output(x, attn, frames_num)
    
    def _finalize_output(self, x, attn, frames_num):
        """
        Finalizza output del transformer.
        
        Identico a HTSAT originale: gestisce TSCAM, pooling, classificazione.
        Non modificato per mantenere compatibilità.
        """
        if self.config.enable_tscam:
            # TSCAM mode: time-frequency attention + classification
            x = self.norm(x)
            # x: (B, H_final * W_final, C_final)
            B, N, C = x.shape
            # B: batch size, N = H_final * W_final, C = C_final

            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            # SF: numero di frame spaziali sull'asse frequenza dopo tutti i layer
            # ST: numero di frame spaziali sull'asse tempo dopo tutti i layer

            x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
            # x: (B, C_final, SF, ST)
            B, C, F, T = x.shape
            # F = SF (asse frequenza), T = ST (asse tempo)

            c_freq_bin = F // self.freq_ratio
            # c_freq_bin: numero di bin frequenziali per gruppo TSCAM
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            # x: (B, C_final, freq_ratio, c_freq_bin, T)
            x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            # dopo permute: (B, C_final, c_freq_bin, freq_ratio, T)
            # dopo reshape: (B, C_final, c_freq_bin, freq_ratio * T)
            
            latent_output = self.avgpool(torch.flatten(x, 2))
            # torch.flatten(x, 2): (B, C_final, c_freq_bin * freq_ratio * T)
            # avgpool:             (B, C_final, 1)
            latent_output = torch.flatten(latent_output, 1)
            # latent_output: (B, C_final)

            if self.config.htsat_attn_heatmap:
                attn = torch.mean(attn, dim=1)
                # attn prima: (B_, num_heads, N, N)
                # dopo mean:  (B_, N, N)
                attn = torch.mean(attn, dim=1)
                # attn: (B_, N)
                attn = attn.reshape(B, SF, ST)
                # attn: (B, SF, ST)
                c_freq_bin = SF // self.freq_ratio
                attn = attn.reshape(B, SF // c_freq_bin, c_freq_bin, ST)
                # attn: (B, freq_ratio, c_freq_bin, ST)
                attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, c_freq_bin, -1)
                # dopo permute: (B, c_freq_bin, freq_ratio, ST)
                # dopo reshape: (B, c_freq_bin, freq_ratio * ST)
                attn = attn.mean(dim=1)
                # attn: (B, freq_ratio * ST)
                attn_max = torch.max(attn, dim=1, keepdim=True)[0]
                attn_min = torch.min(attn, dim=1, keepdim=True)[0]
                # attn_max, attn_min: (B, 1)   [per normalizzazione]
                attn = ((attn * 0.15) + (attn_max * 0.85 - attn_min)) / (attn_max - attn_min)
                # attn: (B, freq_ratio * ST)   [normalizzata]
                attn = attn.unsqueeze(dim=2)
                # attn: (B, freq_ratio * ST, 1)
            
            x = self.tscam_conv(x)
            # x prima:  (B, C_final, c_freq_bin, freq_ratio * T)
            # x dopo:   (B, classes_num, c_freq_bin, freq_ratio * T)   [conv 1x1]
            x = torch.flatten(x, 2)
            # x: (B, classes_num, c_freq_bin * freq_ratio * T)

            if self.config.htsat_attn_heatmap:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous() * attn, 8 * self.patch_stride[1])
                # torch.sigmoid(x):         (B, classes_num, c_freq_bin * freq_ratio * T)
                # .permute(0,2,1):          (B, c_freq_bin * freq_ratio * T, classes_num)
                # * attn:                   (B, c_freq_bin * freq_ratio * T, classes_num)  [broadcast]
                # interpolate(..., factor): (B, c_freq_bin * freq_ratio * T * factor, classes_num)
                # fpx:                      (B, ST_final, classes_num)   [framewise output]
            else:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
                # torch.sigmoid(x):         (B, classes_num, c_freq_bin * freq_ratio * T)
                # .permute(0,2,1):          (B, c_freq_bin * freq_ratio * T, classes_num)
                # interpolate(..., factor): (B, ST_final, classes_num)
            
            x = self.avgpool(x)
            # x prima:  (B, classes_num, c_freq_bin * freq_ratio * T)
            # x dopo:   (B, classes_num, 1)
            x = torch.flatten(x, 1)
            # x: (B, classes_num)

            if self.config.loss_type == "clip_ce":
                output_dict = {
                    'framewise_output': fpx,           # (B, ST_final, classes_num)
                    'clipwise_output': x,              # (B, classes_num)   [logits raw]
                    'latent_output': latent_output     # (B, C_final)
                }
            else:
                output_dict = {
                    'framewise_output': fpx,                   # (B, ST_final, classes_num)
                    'clipwise_output': torch.sigmoid(x),       # (B, classes_num)   [probabilità]
                    'latent_output': latent_output             # (B, C_final)
                }
        else:
            # Standard mode: simple pooling + classification
            x = self.norm(x)
            # x: (B, H_final * W_final, C_final)
            B, N, C = x.shape
            
            fpx = x.permute(0, 2, 1).contiguous().reshape(
                B, C, 
                frames_num // (2 ** (len(self.depths) + 1)), 
                frames_num // (2 ** (len(self.depths) + 1))
            )
            # x.permute(0,2,1):  (B, C_final, H_final * W_final)
            # dopo reshape:      (B, C_final, SF, ST)   [ricostruzione spaziale]
            B, C, F, T = fpx.shape
            # F = SF (frequenza), T = ST (tempo)
            
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            # fpx: (B, C_final, freq_ratio, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            # dopo permute: (B, C_final, c_freq_bin, freq_ratio, T)
            # dopo reshape: (B, C_final, c_freq_bin, freq_ratio * T)
            fpx = torch.sum(fpx, dim=2)
            # fpx: (B, C_final, freq_ratio * T)   [somma sui bin di frequenza]
            
            from .pytorch_utils import interpolate
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            # fpx.permute(0,2,1):   (B, freq_ratio * T, C_final)
            # interpolate:          (B, freq_ratio * T * factor, C_final)   [framewise]
            
            x = self.avgpool(x.transpose(1, 2))
            # x.transpose(1,2):   (B, C_final, H_final * W_final)
            # avgpool:             (B, C_final, 1)
            x = torch.flatten(x, 1)
            # x: (B, C_final)
            
            if self.num_classes > 0:
                x = self.head(x)
                # x:   (B, C_final) → (B, classes_num)
                fpx = self.head(fpx)
                # fpx: (B, ST_final, C_final) → (B, ST_final, classes_num)
            
            output_dict = {
                'framewise_output': torch.sigmoid(fpx),   # (B, ST_final, classes_num)
                'clipwise_output': torch.sigmoid(x)       # (B, classes_num)
            }
        
        return output_dict
    
    def _average_outputs(self, output_dicts, device, include_latent=False):
        """
        Media gli output di multiple forward pass.
        
        Usato quando si processano audio lunghi con overlapping crops.
        
        Args:
            output_dicts: Lista di dizionari di output
            device: Device per i tensori
            include_latent: Se True, media anche latent_output
            
        Returns:
            Dizionario con output mediati
        """
        result = {}
        keys = ['clipwise_output', 'framewise_output', 'embedding']
        if include_latent:
            keys.append('latent_output')
        
        # Filtra solo chiavi disponibili
        available_keys = [k for k in keys if k in output_dicts[0]]
        
        # Media ogni chiave
        for key in available_keys:
            stacked = torch.stack([d[key] for d in output_dicts])
            # stacked: (num_crops, B, ...)   [una entry per crop]
            result[key] = stacked.mean(dim=0)
            # result[key]: (B, ...)   [media sui crop]
        
        return result
    
    def fit_spectral_layers(self, dataloader, max_samples: int = 10000):
        """
        Raccoglie dati e calcola PCA per tutti gli spectral layers.
        
        Pipeline:
        1. RACCOLTA DATI:
           - Mode 'attention': raccoglie output di ogni testa per ogni layer target
           - Mode 'layer': raccoglie output completo di ogni layer target
        
        2. FIT PCA:
           - Mode 'attention': fit separato per ogni testa
           - Mode 'layer': fit per layer completo
        
        Args:
            dataloader: DataLoader con dati audio per fitting
            max_samples: Numero massimo di samples da raccogliere
        
        Returns:
            variance_info: Dizionario con % varianza spiegata per ogni component
        """
        self.eval()

        # Clear dictionary to store collected data
        self.collected_data.clear()
        
        print(f"\n{'='*80}")
        print(f"📊 Fitting Spectral Layers")
        print(f"{'='*80}")
        print(f"Modalità: {self.mode.upper()}")
        print(f"Target layers: {self.target_layers}")
        print(f"Max samples: {max_samples}")
        print(f"{'='*80}\n")
        
        # ========== FASE 1: RACCOLTA DATI ==========
        print("📦 Fase 1: Raccolta hidden states...")
        n_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Raccolta")
            
            for batch in pbar:
                if n_samples >= max_samples:
                    break
                
                # Estrai audio dal batch (gestisce vari formati)
                if isinstance(batch, dict):
                    audio = batch.get('audio', batch.get('waveform'))
                else:
                    audio = batch[0] if isinstance(batch, (list, tuple)) else batch
                # audio: (B, audio_samples)

                # Sposta su GPU se necessario
                if next(self.parameters()).is_cuda and not audio.is_cuda:
                    audio = audio.cuda()
                
                # Forward con flag di raccolta
                try:
                    _ = self.forward(audio, collect_for_fitting=True)
                    n_samples += audio.size(0)
                    # audio.size(0) = B (sample nel batch corrente)
                    pbar.set_postfix({'samples': n_samples})
                    
                except Exception as e:
                    print(f"\n⚠️  Errore durante forward: {e}")
                    continue
        
        print(f"\n✓ Raccolti {n_samples} samples totali")
        
        # ========== FASE 2: FIT PCA ==========
        print(f"\n{'='*80}")
        print("🔬 Fase 2: Calcolo PCA e inizializzazione pesi...")
        print(f"{'='*80}\n")
        
        variance_info = {}
        
        if self.mode == 'attention':
            # MODE ATTENTION: Fit PCA per ogni testa di ogni blocco
            
            for layer_name in self.collected_data.keys():
                print(f"\n{layer_name}:")
                variance_info[layer_name] = {}
                
                # Ottieni ModuleList di ModuleList: [block][head] -> SpectralLayer
                block_spectral_layers = self.spectral_layers[layer_name]
                
                # Itera su ogni blocco
                for block_idx in range(len(block_spectral_layers)):
                    print(f"\n  Block {block_idx}:")
                    variance_info[layer_name][f'block_{block_idx}'] = {}
                    
                    spectral_layers_for_block = block_spectral_layers[block_idx]
                    
                    # Fit PCA per ogni testa del blocco
                    for head_idx, spectral_layer in enumerate(spectral_layers_for_block):
                        head_name = f'head_{head_idx}'
                        head_data_list = self.collected_data[layer_name][block_idx][head_name]
                        # head_data_list: lista di tensori (B_ * N, head_dim) per ogni batch
                        
                        if not head_data_list:
                            print(f"    ⚠️  {head_name}: nessun dato, skip")
                            continue

                        # Concatena tutti i batch raccolti
                        combined = torch.cat(head_data_list, dim=0)
                        # combined: (N_pca_total, head_dim)
                        #   N_pca_total = somma su tutti i batch di (B_ * N)
                        
                        # Fit PCA e ottieni varianza spiegata
                        variance_ratio = spectral_layer.fit_pca(combined)
                        # combined:       (N_pca_total, head_dim)   → input a PCA
                        # variance_ratio: (n_comp,)                 → % varianza per PC
                        variance_info[layer_name][f'block_{block_idx}'][head_name] = variance_ratio
                        
                        # Stampa info compatta
                        print(f"    ✓ {head_name}: shape={combined.shape}, "
                              f"var={[f'{v:.3f}' for v in variance_ratio[:3]]}, "
                              f"cum={variance_ratio[:3].sum():.3f}")
        
        else:  # mode == 'layer'
            # MODE LAYER: Fit PCA per layer completo
            
            for layer_name, layer_data_list in self.collected_data.items():
                if not layer_data_list:
                    print(f"⚠️  {layer_name}: nessun dato, skip")
                    continue
                
                # Concatena tutti i batch raccolti
                combined = torch.cat(layer_data_list, dim=0)
                # layer_data_list: lista di tensori (B * H_out * W_out, C_out) per ogni batch
                # combined:        (N_pca_total, C_out)
                #   N_pca_total = somma su tutti i batch di (B * H_out * W_out)
                
                # Fit PCA
                variance_ratio = self.spectral_layers[layer_name].fit_pca(combined)
                # combined:       (N_pca_total, C_out) → input a PCA
                # variance_ratio: (n_comp,)            → % varianza per PC
                variance_info[layer_name] = variance_ratio
                
                # Stampa info
                print(f"\n✓ {layer_name}:")
                print(f"  Shape: {combined.shape}")
                print(f"  Top-5 variance: {[f'{v:.4f}' for v in variance_ratio[:5]]}")
                print(f"  Cumulative top-5: {variance_ratio[:5].sum():.4f}")
                print(f"  Total explained: {variance_ratio.sum():.4f}")
        
        # Pulizia memoria
        self.collected_data = {}
        
        print(f"\n{'='*80}")
        print("✅ Fitting completato!")
        print(f"{'='*80}\n")
        
        return variance_info

# ============================================================================
# WRAPPERS - Per integrazione con CLAP
# ============================================================================

class ResiDualHTSATWrapper(nn.Module):
    """
    Wrapper per ResiDualHTSAT che aggiunge compatibilità con CLAP.
    
    Converte l'output di ResiDualHTSAT nel formato atteso da CLAP.
    """
    
    def __init__(self, residual_config=None):
        super().__init__()
        from . import config

        if residual_config is None:
            raise ValueError("residual_config è obbligatorio")
        
        self.htsat = ResiDualHTSAT(config=config, residual_config=residual_config)

    def forward(self, x):
        """
        Forward che aggiunge campo 'embedding' per compatibilità CLAP.

        Args:
            x: (B, audio_samples)

        Returns:
            out_dict con 'embedding': (B, C_final), 'clipwise_output': (B, classes_num), etc.
        """
        out_dict = self.htsat(x)
        # CLAP si aspetta 'embedding', mappiamo da 'latent_output'
        # latent_output: (B, C_final)
        out_dict['embedding'] = out_dict['latent_output']
        # embedding: (B, C_final)
        return out_dict


class AudioEncoder(nn.Module):
    """
    Audio encoder per Residual CLAP con projection layer.
    
    Pipeline:
    1. ResiDualHTSAT estrae features audio
    2. Projection mappa features allo spazio multimodale
    """
    
    def __init__(self, audioenc_name: str, d_in: int, d_out: int, sample_rate: int, 
                 window_size: int, hop_size: int, mel_bins: int, fmin: int, fmax: int, 
                 classes_num: int, residual_config: Dict) -> None:
        super().__init__()
        self.base = ResiDualHTSATWrapper(residual_config=residual_config)
        self.projection = Projection(d_in, d_out)
        # Projection: (B, d_in) → (B, d_out)
        #   d_in:  C_final (dimensione embedding finale di HTSAT)
        #   d_out: dimensione dello spazio multimodale condiviso audio-testo

    def forward(self, x):
        """
        Forward con projection.

        Args:
            x: (B, audio_samples)

        Returns:
            projected:              (B, d_out)      features nello spazio multimodale
            audio_classification:  (B, classes_num) probabilità di classificazione
        """
        out_dict = self.base(x)
        audio_features = out_dict['embedding']
        # audio_features: (B, C_final)   [embedding grezzo di HTSAT]
        audio_classification = out_dict['clipwise_output']
        # audio_classification: (B, classes_num)
        projected = self.projection(audio_features)
        # projected: (B, d_out)   [proiettato nello spazio multimodale]
        return projected, audio_classification


class ResiDualCLAP(CLAP):
    """
    CLAP con ResiDual spectral reweighting.
    
    Estende CLAP standard sostituendo l'audio encoder con ResiDualHTSAT.
    Mantiene compatibilità completa con l'API CLAP originale.

    """
    
    def __init__(self, *args, residual_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)

        if residual_config is None:
            raise ValueError("residual_config è obbligatorio")
        
        self.residual_config = residual_config
        
        # Versione ResiDual
        self.audio_encoder = AudioEncoder(
            kwargs["audioenc_name"], 
            kwargs["out_emb"],    # d_in:  dimensione embedding HTSAT (es. 768)
            kwargs["d_proj"],     # d_out: dimensione spazio multimodale (es. 512)
            kwargs["sample_rate"], 
            kwargs["window_size"], 
            kwargs["hop_size"], 
            kwargs["mel_bins"],
            kwargs["fmin"], 
            kwargs["fmax"], 
            kwargs["classes_num"],
            self.residual_config
        )

    def fit_spectral_components(self, audio_dataloader, max_samples=10000):
        """
        Wrapper per fit_spectral_layers del ResiDualHTSAT interno.
        
        Args:
            audio_dataloader: DataLoader con audio (B, audio_samples) per fitting PCA
            max_samples: Numero massimo di samples da usare
            
        Returns:
            variance_info: Dizionario con informazioni sulla varianza spiegata
        """
        if hasattr(self.audio_encoder.base.htsat, 'fit_spectral_layers'):
            return self.audio_encoder.base.htsat.fit_spectral_layers(
                audio_dataloader, 
                max_samples
            )
        return {}