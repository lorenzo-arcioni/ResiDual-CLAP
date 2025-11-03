# models/residual_clap.py - VERSIONE CORRETTA PER CODEBASE ORIGINALE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from .clap import CLAP, Projection
from .htsat import HTSAT_Swin_Transformer


class SpectralReweightingLayer(nn.Module):
    """
    Layer che applica riweighting spettrale basato su PCA
    """
    def __init__(self, embed_dim: int, n_components: int = None, reweight_factor: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_components = n_components or embed_dim // 4
        self.reweight_factor = reweight_factor
        
        # Pesi apprendibili per ogni componente principale
        self.pc_weights = nn.Parameter(torch.ones(self.n_components))
        
        # Buffer per componenti PCA
        self.register_buffer('pca_components', torch.eye(embed_dim, self.n_components))
        self.register_buffer('pca_mean', torch.zeros(embed_dim))
        self.register_buffer('is_fitted', torch.tensor(False))
        
    def fit_pca(self, head_outputs: torch.Tensor):
        """Fit PCA su outputs raccolti"""
        X = head_outputs.detach().cpu().numpy()
        
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        
        self.pca_components.data = torch.tensor(pca.components_.T, dtype=torch.float32)
        self.pca_mean.data = torch.tensor(pca.mean_, dtype=torch.float32)
        self.is_fitted.data = torch.tensor(True)
        
        return pca.explained_variance_ratio_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applica spectral reweighting"""
        if not self.is_fitted:
            return x
            
        original_shape = x.shape
        x_flat = x.view(-1, self.embed_dim)
        
        # Centra i dati
        x_centered = x_flat - self.pca_mean
        
        # Proietta su PC
        pc_projections = torch.matmul(x_centered, self.pca_components)
        
        # Applica reweighting
        weighted_projections = pc_projections * self.pc_weights
        
        # Ricostruisci
        reconstructed = torch.matmul(weighted_projections, self.pca_components.T)
        
        # Aggiungi residuo
        residual = x_centered - torch.matmul(pc_projections, self.pca_components.T)
        reweighted = reconstructed + residual + self.pca_mean
        
        return reweighted.view(original_shape)


class ResiDualHTSAT(HTSAT_Swin_Transformer):
    """
    HTSAT con spectral reweighting layers
    """
    def __init__(self, *args, residual_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.residual_config = residual_config or {
            'n_components_ratio': 0.25,
            'reweight_factor': 2.0,
            'target_layers': [1, 3],
            'analysis_mode': False
        }
        
        self._add_spectral_layers()
        self.head_outputs = []
        
    def _add_spectral_layers(self):
        """Aggiungi spectral layers ai target layers - VERSIONE CORRETTA"""
        target_layers = self.residual_config.get('target_layers', [])
        n_components_ratio = self.residual_config.get('n_components_ratio', 0.25)
        
        self.spectral_layers = nn.ModuleDict()
        
        # PRIMA: Fai un forward pass per scoprire le VERE dimensioni
        print("🔍 Detecting actual layer dimensions...")
        self.residual_config['analysis_mode'] = True
        self.head_outputs = []
        
        # Forward pass con audio di test
        with torch.no_grad():
            test_audio = torch.randn(1, self.config.clip_samples if hasattr(self.config, 'clip_samples') else 320000)
            if next(self.parameters()).is_cuda:
                test_audio = test_audio.cuda()
            
            try:
                _ = self.forward(test_audio)
            except:
                pass  # Ignora eventuali errori, ci interessano solo le dimensioni
        
        # Crea spectral layers con dimensioni REALI
        for layer_idx in target_layers:
            # Cerca l'output di questo layer
            layer_output = None
            for output_dict in self.head_outputs:
                if output_dict['layer'] == layer_idx:
                    layer_output = output_dict['output']
                    break
            
            if layer_output is not None:
                # Usa la dimensione REALE dall'output
                actual_dim = layer_output.shape[-1]
                layer_n_components = int(actual_dim * n_components_ratio)
                
                layer_name = f'layer_{layer_idx}'
                self.spectral_layers[layer_name] = SpectralReweightingLayer(
                    embed_dim=actual_dim,  # ← USA DIMENSIONE REALE
                    n_components=layer_n_components,
                    reweight_factor=self.residual_config.get('reweight_factor', 2.0)
                )
                print(f"  ✓ {layer_name}: {actual_dim} dims → {layer_n_components} components")
            else:
                print(f"  ⚠️  Could not detect dimensions for layer {layer_idx}")
        
        self.residual_config['analysis_mode'] = False
        self.head_outputs = []
    
    def forward(self, x, mixup_lambda=None, infer_mode=False):
        """Forward con spectral reweighting - VERSIONE SEMPLIFICATA"""
        
        # Preprocessing standard HTSAT
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
        
        # Reshape per HTSAT
        x = self.reshape_wav2img(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # Forward attraverso layers con reweighting
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            
            # Colleziona outputs se in analysis mode
            if self.residual_config.get('analysis_mode', False):
                self.head_outputs.append({
                    'layer': i,
                    'output': x.detach().clone(),
                    'attention': attn.detach().clone() if attn is not None else None
                })
            
            # Applica spectral reweighting
            layer_name = f'layer_{i}'
            if layer_name in self.spectral_layers:
                x = self.spectral_layers[layer_name](x)
        
        # Normalization
        x = self.norm(x)
        
        # Pooling per ottenere embedding
        # Usa mean pooling semplice
        x_pooled = x.mean(dim=1)  # [B, N, C] -> [B, C]
        
        # Output dict - semplificato
        output_dict = {
            'embedding': x_pooled,
            'clipwise_output': None  # Non necessario per il nostro caso
        }
        
        return output_dict
    
    def fit_spectral_layers(self, dataloader, max_samples: int = 10000):
        """Fit PCA per ogni spectral layer"""
        self.eval()
        self.residual_config['analysis_mode'] = True
        
        print("Collecting head outputs for PCA fitting...")
        
        collected_outputs = {layer_name: [] for layer_name in self.spectral_layers.keys()}
        n_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if n_samples >= max_samples:
                    break
                
                if isinstance(batch, dict):
                    audio = batch.get('audio', batch.get('waveform'))
                else:
                    audio = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                self.head_outputs = []
                _ = self.forward(audio)
                
                for output_dict in self.head_outputs:
                    layer_idx = output_dict['layer']
                    layer_name = f'layer_{layer_idx}'
                    
                    if layer_name in collected_outputs:
                        output_tensor = output_dict['output']
                        flattened = output_tensor.view(-1, output_tensor.size(-1))
                        collected_outputs[layer_name].append(flattened)
                
                n_samples += audio.size(0)
        
        # Fit PCA
        variance_ratios = {}
        for layer_name, outputs in collected_outputs.items():
            if outputs:
                combined_outputs = torch.cat(outputs, dim=0)
                print(f"Fitting PCA for {layer_name} with {combined_outputs.shape[0]} samples...")
                
                variance_ratio = self.spectral_layers[layer_name].fit_pca(combined_outputs)
                variance_ratios[layer_name] = variance_ratio
                
                print(f"{layer_name}: Top 5 PC variance ratios: {variance_ratio[:5]}")
        
        self.residual_config['analysis_mode'] = False
        return variance_ratios


class ResiDualCLAP(nn.Module):
    """
    CLAP con ResiDual spectral reweighting
    VERSIONE SEMPLIFICATA che riusa AudioEncoder originale
    """
    def __init__(self, audioenc_name: str, sample_rate: int, window_size: int,
                 hop_size: int, mel_bins: int, fmin: int, fmax: int,
                 classes_num: int, out_emb: int, text_model: str,
                 transformer_embed_dim: int, d_proj: int,
                 residual_config: Dict = None):
        
        super().__init__()
        
        self.residual_config = residual_config or {}
        
        # ========================================
        # CAMBIAMENTO: Usa AudioEncoder originale
        # ========================================
        from .clap import AudioEncoder  # ← Import l'originale
        
        # Crea AudioEncoder standard
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num
        )
        
        # Sostituisci solo la base con ResiDualHTSAT
        if audioenc_name == "HTSAT":
            from . import config
            self.audio_encoder.base = ResiDualHTSAT(
                config=config, 
                residual_config=residual_config
            )
        else:
            raise ValueError(f"ResiDual currently only supports HTSAT, not {audioenc_name}")
        
        # Text encoder (standard CLAP)
        from .clap import TextEncoder
        self.caption_encoder = TextEncoder(d_proj, text_model, transformer_embed_dim)
        
        # Logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, audio, text):
        """Forward standard CLAP"""
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)
        return caption_embed, audio_embed, self.logit_scale.exp()
    
    def fit_spectral_components(self, audio_dataloader, max_samples: int = 10000):
        """Fit spectral components pubblico"""
        if hasattr(self.audio_encoder.base, 'fit_spectral_layers'):
            return self.audio_encoder.base.fit_spectral_layers(audio_dataloader, max_samples)
        else:
            print("Audio encoder non supporta spectral fitting")
            return {}