"""
ResiDual HTSAT: Spectral Reweighting per Audio Transformer

Due modalità operative:
1. ATTENTION MODE: Reweighting per-head (ogni testa di attenzione ha il suo PCA)
2. LAYER MODE: Reweighting per-layer (un PCA per l'output completo del layer)
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
    Layer che applica reweighting spettrale tramite PCA.
    
    Pipeline:
    1. fit_pca(data): Calcola PCA su dati raccolti
       - Trova componenti principali che catturano la varianza
       - Inizializza pesi proporzionali alla varianza spiegata
    
    2. forward(x): Applica reweighting
       - Centra i dati (x - mean)
       - Proietta sulle componenti principali
       - Amplifica le componenti pesate
       - Ricostruisce + aggiunge residuo
    """
    
    def __init__(self, embed_dim: int, n_components: int = None, reweight_factor: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_components = n_components or embed_dim // 4  # Default: 25% delle dimensioni
        self.reweight_factor = reweight_factor
        
        # Pesi appresi (trainable): uno per ogni componente principale
        # Inizializzati a 1.0, poi aggiustati in base alla varianza durante fit
        self.pc_weights = nn.Parameter(torch.ones(self.n_components))
        
        # Buffers PCA (non-trainable): salvano i risultati del fit
        self.register_buffer('pca_components', torch.eye(embed_dim, self.n_components))
        self.register_buffer('pca_mean', torch.zeros(embed_dim))
        self.register_buffer('is_fitted', torch.tensor(False))
        
    def fit_pca(self, data: torch.Tensor):
        """
        Calcola PCA sui dati raccolti e inizializza i pesi.
        
        Args:
            data: Tensor (N, embed_dim) con N samples raccolti durante training
        
        Returns:
            variance_ratio: Array con % di varianza spiegata da ogni PC
        """
        X = data.detach().cpu().numpy()
        
        # Calcola PCA con scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        
        # Salva componenti principali (vettori che definiscono il nuovo spazio)
        # Shape: (embed_dim, n_components)
        self.pca_components.data = torch.tensor(pca.components_.T, dtype=torch.float32)
        
        # Salva media per centrare i dati
        self.pca_mean.data = torch.tensor(pca.mean_, dtype=torch.float32)
        
        # Inizializza pesi basandosi sulla varianza spiegata
        # Le componenti con più varianza ricevono pesi maggiori
        variance_ratio = pca.explained_variance_ratio_
        self.pc_weights.data = torch.tensor(
            1.0 + self.reweight_factor * variance_ratio,
            dtype=torch.float32
        )
        
        self.is_fitted.data = torch.tensor(True)
        return variance_ratio
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applica spectral reweighting.
        
        Steps:
        1. Centra i dati: x_centered = x - mean
        2. Proietta su PC: coefficienti = x_centered @ components
        3. Applica pesi: weighted_coef = coefficienti * weights
        4. Ricostruisci: reconstructed = weighted_coef @ components^T
        5. Aggiungi residuo (parte non catturata da PC)
        6. De-centra: result = reconstructed + residuo + mean
        
        Args:
            x: Input tensor di qualsiasi shape (..., embed_dim)
        
        Returns:
            Tensor reweighted con stessa shape di x
        """
        if not self.is_fitted:
            # Se PCA non è stata fatta, passa through
            return x
            
        original_shape = x.shape
        x_flat = x.reshape(-1, self.embed_dim)  # ✅ CAMBIATO: reshape invece di view
        
        # Step 1: Centra i dati
        x_centered = x_flat - self.pca_mean
        
        # Step 2: Proietta sulle componenti principali
        pc_proj = torch.matmul(x_centered, self.pca_components)
        
        # Step 3: Applica pesi appresi (amplifica componenti importanti)
        weighted_proj = pc_proj * self.pc_weights
        
        # Step 4: Ricostruisci nello spazio originale
        reconstructed = torch.matmul(weighted_proj, self.pca_components.T)
        
        # Step 5: Calcola residuo (informazione non catturata dalle PC)
        residual = x_centered - torch.matmul(pc_proj, self.pca_components.T)
        
        # Step 6: Output finale = ricostruzione + residuo + media
        result = reconstructed + residual + self.pca_mean
        
        return result.reshape(original_shape)


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
    4. 🔥 RACCOLTA (opzionale): salva head_i in collected_data
    5. 🔥 REWEIGHTING: head_i_reweighted = spectral_layer_i(head_i)
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
            x: Input (num_windows*B, N, C)
            mask: Attention mask opzionale
            collect_for_fitting: Se True, salva head outputs in collected_data
            
        Returns:
            x: Output reweighted (num_windows*B, N, C)
            attn: Attention weights per visualizzazione
        """
        B_, N, C = x.shape
        
        # Step 1: Calcola Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Step 2: Calcola attention scores
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Step 3: Aggiungi relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Step 4: Applica mask se presente
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Step 5: Calcola output attention per ogni testa
        x_heads = attn @ v  # (B_, num_heads, N, head_dim)
        
        # Step 6: RACCOLTA DATI (se richiesto)
        if collect_for_fitting and self.collected_data is not None:
            self._collect_heads(x_heads)
        
        # Step 7: REWEIGHTING PER-HEAD (se fitted)
        if self.spectral_layers is not None:
            reweighted_heads = []
            for head_idx in range(self.num_heads):
                head_output = x_heads[:, head_idx, :, :]
                reweighted = self.spectral_layers[head_idx](head_output)
                reweighted_heads.append(reweighted)
            x_heads = torch.stack(reweighted_heads, dim=1)
        
        # Step 8: Concatena teste e applica projection finale
        x = x_heads.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
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
            head_data = x_heads[:, head_idx, :, :]  # (B_, N, head_dim)
            reshaped = head_data.reshape(-1, head_data.size(-1))
            self.collected_data[layer_name][self.block_idx][f'head_{head_idx}'].append(
                reshaped.detach().cpu()
            )


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
        reweight_factor = self.residual_config.get('reweight_factor', 2.0)
        
        print(f"\n{'='*80}")
        print(f"🔧 Setup ResiDual HTSAT")
        print(f"{'='*80}")
        print(f"Modalità: {self.mode.upper()}")
        print(f"Target layers: {self.target_layers}")
        print(f"PCA components ratio: {n_components_ratio}")
        print(f"Reweight factor: {reweight_factor}")
        
        for layer_idx in self.target_layers:
            if layer_idx >= len(self.layers):
                print(f"⚠️  Layer {layer_idx} non esiste (max: {len(self.layers)-1}), skip")
                continue
            
            # Calcola dimensione del layer (struttura Swin: raddoppia ogni layer)
            layer_dim = int(self.embed_dim * (2 ** layer_idx))
            layer_name = f'layer_{layer_idx}'
            
            if self.mode == 'attention':
                # MODE ATTENTION: Crea uno spectral layer per ogni testa di ogni blocco
                num_heads = self.num_heads[layer_idx]
                num_blocks = self.depths[layer_idx]
                head_dim = layer_dim // num_heads
                n_components = int(head_dim * n_components_ratio)
                
                # Crea ModuleList di ModuleList: [blocco][testa]
                # Ogni blocco ha le sue teste, ognuna con il proprio SpectralLayer
                block_spectral_layers = nn.ModuleList([
                    nn.ModuleList([
                        SpectralReweightingLayer(
                            embed_dim=head_dim,
                            n_components=n_components,
                            reweight_factor=reweight_factor
                        )
                        for _ in range(num_heads)
                    ])
                    for _ in range(num_blocks)
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
                n_components = int(layer_dim * n_components_ratio)
                
                self.spectral_layers[layer_name] = SpectralReweightingLayer(
                    embed_dim=layer_dim,
                    n_components=n_components,
                    reweight_factor=reweight_factor
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
        """
        target_layer = self.layers[layer_idx]
        
        for block_idx, block in enumerate(target_layer.blocks):
            old_attn = block.attn
            spectral_layers_for_block = block_spectral_layers[block_idx]
            
            # MODIFICATO: Estratti dropout rates in modo sicuro
            attn_drop_rate = old_attn.attn_drop.p if hasattr(old_attn.attn_drop, 'p') else 0.
            proj_drop_rate = old_attn.proj_drop.p if hasattr(old_attn.proj_drop, 'p') else 0.
            
            # Crea nuova attention con metadati per raccolta
            # MODIFICATO: qk_scale=None invece di old_attn.qk_scale (che non esiste)
            block.attn = WindowAttentionReweighting(
                dim=old_attn.dim,
                window_size=old_attn.window_size,
                num_heads=old_attn.num_heads,
                qkv_bias=True,
                qk_scale=None,  # Verrà calcolato automaticamente come head_dim ** -0.5
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                spectral_layers=spectral_layers_for_block,
                layer_idx=layer_idx,
                block_idx=block_idx,
                collected_data=self.collected_data  # 🔥 Riferimento condiviso
            )
            
            # Copia pesi dal vecchio modulo
            block.attn.load_state_dict(old_attn.state_dict(), strict=False)
        
    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False, collect_for_fitting=False):
        """
        Forward pass con opzione di raccolta dati per fitting PCA.
        
        Args:
            x: Input audio waveform
            mixup_lambda: Coefficiente per mixup durante training
            infer_mode: Se True, gestisce input di lunghezza variabile
            collect_for_fitting: Se True, salva hidden states per fitting PCA
        
        Returns:
            output_dict: Dizionario con clipwise_output, framewise_output, etc.
        """
        
        # ========== PREPROCESSING AUDIO ==========
        # Identico a HTSAT originale: spectrogram -> logmel -> normalization
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Augmentation durante training
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            from .pytorch_utils import do_mixup
            x = do_mixup(x, mixup_lambda)
        
        # ========== GESTIONE VARI INPUT MODE ==========
        # Logica identica a HTSAT originale per compatibilità
        
        if infer_mode:
            # Inference mode: ripeti spectrogram per raggiungere target size
            frame_num = x.shape[2]
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
            x = self.reshape_wav2img(x)
            output_dict = self._forward_features(x, collect_for_fitting)
        
        elif self.config.enable_repeat_mode:
            # Repeat mode: processa multiple posizioni e media
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                output_dict = self._forward_features(x, collect_for_fitting)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    tx = self.repeat_wat2img(tx, cur_pos)
                    output_dicts.append(self._forward_features(tx, False))
                output_dict = self._average_outputs(output_dicts, x.device)
        
        else:
            # Standard mode
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    # Training: random crop
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    x = self.reshape_wav2img(x)
                    output_dict = self._forward_features(x, collect_for_fitting)
                else:
                    # Inference: overlapping crops + media
                    overlap_size = 344
                    crop_size = 689
                    output_dicts = []
                    
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
                        tx = self.reshape_wav2img(tx)
                        output_dicts.append(self._forward_features(tx, False))
                    
                    output_dict = self._average_outputs(output_dicts, x.device, include_latent=True)
            else:
                x = self.reshape_wav2img(x)
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
            x: Input dopo preprocessing (B, C, H, W)
            collect_for_fitting: Flag per raccolta dati PCA
            
        Returns:
            output_dict: Clipwise/framewise outputs
        """
        frames_num = x.shape[2]

        # ========== PATCH EMBEDDING ==========
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # ========== TRANSFORMER LAYERS ==========
        for i, layer in enumerate(self.layers):
            layer_name = f'layer_{i}'
            
            if self.mode == 'attention' and layer_name in self.spectral_layers and collect_for_fitting:
                # MODE ATTENTION con FITTING: Forward manuale dei blocchi con flag di raccolta
                attns = []
                for block_idx, block in enumerate(layer.blocks):
                    if layer.use_checkpoint:
                        from torch.utils import checkpoint
                        x = checkpoint.checkpoint(block, x)
                    else:
                        # Forward completo del blocco (non solo attention)
                        shortcut = x
                        x = block.norm1(x)
                        x_shape = x.shape
                        
                        # Reshape to 4D
                        x = x.reshape(x_shape[0], block.input_resolution[0], block.input_resolution[1], x_shape[2])
                        
                        # Cyclic shift
                        if block.shift_size > 0:
                            shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                        else:
                            shifted_x = x
                        
                        # Window partition
                        from .htsat import window_partition, window_reverse
                        x_windows = window_partition(shifted_x, block.window_size)
                        x_windows = x_windows.reshape(-1, block.window_size * block.window_size, x_shape[2])
                        
                        # Attention con flag di raccolta
                        attn_windows, attn = block.attn(x_windows, mask=block.attn_mask, collect_for_fitting=True)
                        
                        # Window reverse
                        attn_windows = attn_windows.reshape(-1, block.window_size, block.window_size, x_shape[2])
                        shifted_x = window_reverse(attn_windows, block.window_size, 
                                                block.input_resolution[0], block.input_resolution[1])
                        
                        # Reverse cyclic shift
                        if block.shift_size > 0:
                            x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                        else:
                            x = shifted_x
                        
                        x = x.reshape(x_shape[0], block.input_resolution[0] * block.input_resolution[1], x_shape[2])
                        
                        # FFN
                        x = shortcut + block.drop_path(x)
                        x = x + block.drop_path(block.mlp(block.norm2(x)))
                        
                        if not self.training:
                            attns.append(attn.unsqueeze(0))
                
                # Downsampling
                if layer.downsample is not None:
                    x = layer.downsample(x)
                
                # Prepara attn
                if not self.training and attns:
                    attn = torch.cat(attns, dim=0)
                    attn = torch.mean(attn, dim=0)
                else:
                    attn = None
            
            elif self.mode == 'layer' and layer_name in self.spectral_layers:
                # MODE LAYER: Forward manuale per intercettare prima del downsampling
                
                # 1. Forward attraverso i blocchi del layer
                attns = []
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        from torch.utils import checkpoint
                        x = checkpoint.checkpoint(blk, x)
                    else:
                        x, attn = blk(x)
                        if not self.training:
                            attns.append(attn.unsqueeze(0))
                
                # 2. Raccogli/applica reweighting PRIMA del downsampling
                if collect_for_fitting:
                    # FITTING: Salva output PRIMA del downsampling
                    if layer_name not in self.collected_data:
                        self.collected_data[layer_name] = []
                    self.collected_data[layer_name].append(
                        x.reshape(-1, x.size(-1)).detach().cpu()
                    )
                else:
                    # INFERENCE: Applica reweighting PRIMA del downsampling
                    x = self.spectral_layers[layer_name](x)
                
                # 3. Applica downsampling se presente
                if layer.downsample is not None:
                    x = layer.downsample(x)
                
                # 4. Prepara attn per output
                if not self.training and attns:
                    attn = torch.cat(attns, dim=0)
                    attn = torch.mean(attn, dim=0)
                else:
                    attn = None
            
            else:
                # Forward normale del layer (senza reweighting o con reweighting già integrato)
                x, attn = layer(x)
        
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
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape

            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            
            latent_output = self.avgpool(torch.flatten(x, 2))
            latent_output = torch.flatten(latent_output, 1)

            if self.config.htsat_attn_heatmap:
                attn = torch.mean(attn, dim=1)
                attn = torch.mean(attn, dim=1)
                attn = attn.reshape(B, SF, ST)
                c_freq_bin = SF // self.freq_ratio
                attn = attn.reshape(B, SF // c_freq_bin, c_freq_bin, ST)
                attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, c_freq_bin, -1)
                attn = attn.mean(dim=1)
                attn_max = torch.max(attn, dim=1, keepdim=True)[0]
                attn_min = torch.min(attn, dim=1, keepdim=True)[0]
                attn = ((attn * 0.15) + (attn_max * 0.85 - attn_min)) / (attn_max - attn_min)
                attn = attn.unsqueeze(dim=2)
            
            x = self.tscam_conv(x)
            x = torch.flatten(x, 2)

            if self.config.htsat_attn_heatmap:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous() * attn, 8 * self.patch_stride[1])
            else:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            if self.config.loss_type == "clip_ce":
                output_dict = {
                    'framewise_output': fpx,
                    'clipwise_output': x,
                    'latent_output': latent_output
                }
            else:
                output_dict = {
                    'framewise_output': fpx,
                    'clipwise_output': torch.sigmoid(x),
                    'latent_output': latent_output
                }
        else:
            # Standard mode: simple pooling + classification
            x = self.norm(x)
            B, N, C = x.shape
            
            fpx = x.permute(0, 2, 1).contiguous().reshape(
                B, C, 
                frames_num // (2 ** (len(self.depths) + 1)), 
                frames_num // (2 ** (len(self.depths) + 1))
            )
            B, C, F, T = fpx.shape
            
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim=2)
            
            from .pytorch_utils import interpolate
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            
            if self.num_classes > 0:
                x = self.head(x)
                fpx = self.head(fpx)
            
            output_dict = {
                'framewise_output': torch.sigmoid(fpx),
                'clipwise_output': torch.sigmoid(x)
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
            result[key] = stacked.mean(dim=0)
        
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

                # Sposta su GPU se necessario
                if next(self.parameters()).is_cuda and not audio.is_cuda:
                    audio = audio.cuda()
                
                # Forward con flag di raccolta
                try:
                    _ = self.forward(audio, collect_for_fitting=True)
                    n_samples += audio.size(0)
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
                        
                        if not head_data_list:
                            print(f"    ⚠️  {head_name}: nessun dato, skip")
                            continue
                        
                        # Concatena tutti i batch raccolti
                        combined = torch.cat(head_data_list, dim=0)
                        
                        # Fit PCA e ottieni varianza spiegata
                        variance_ratio = spectral_layer.fit_pca(combined)
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
                
                # Fit PCA
                variance_ratio = self.spectral_layers[layer_name].fit_pca(combined)
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
        """
        out_dict = self.htsat(x)
        # CLAP si aspetta 'embedding', mappiamo da 'latent_output'
        out_dict['embedding'] = out_dict['latent_output']
        return out_dict


class AudioEncoder(nn.Module):
    """
    Audio encoder per CLAP con projection layer.
    
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

    def forward(self, x):
        """
        Forward con projection.
        
        Returns:
            projected: Features proiettate per contrastive learning
            audio_classification: Output di classificazione audio
        """
        out_dict = self.base(x)
        audio_features = out_dict['embedding']
        audio_classification = out_dict['clipwise_output']
        projected = self.projection(audio_features)
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
        
        # Sostituisci audio encoder con versione ResiDual
        self.audio_encoder = AudioEncoder(
            kwargs["audioenc_name"], 
            kwargs["out_emb"], 
            kwargs["d_proj"],
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
            audio_dataloader: DataLoader con audio per fitting PCA
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