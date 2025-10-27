import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from .clap import CLAP
from .audio import AudioEncoder
from .htsat import HTSAT_Swin_Transformer


class SpectralReweightingLayer(nn.Module):
    """
    Spectral reweighting layer that applies PCA-based reweighting to attention head outputs
    """
    def __init__(self, embed_dim: int, n_components: int = None, reweight_factor: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_components = n_components or embed_dim // 4
        self.reweight_factor = reweight_factor
        
        # Learnable reweighting coefficients for principal components
        self.pc_weights = nn.Parameter(torch.ones(self.n_components))
        
        # PCA components will be stored as buffers
        self.register_buffer('pca_components', torch.eye(embed_dim, self.n_components))
        self.register_buffer('pca_mean', torch.zeros(embed_dim))
        self.register_buffer('is_fitted', torch.tensor(False))
        
    def fit_pca(self, head_outputs: torch.Tensor):
        """
        Fit PCA on collected head outputs
        Args:
            head_outputs: [batch_size * seq_len * n_samples, embed_dim]
        """
        # Convert to numpy for sklearn PCA
        X = head_outputs.detach().cpu().numpy()
        
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        
        # Store PCA components and mean
        self.pca_components.data = torch.tensor(pca.components_.T, dtype=torch.float32)
        self.pca_mean.data = torch.tensor(pca.mean_, dtype=torch.float32)
        self.is_fitted.data = torch.tensor(True)
        
        return pca.explained_variance_ratio_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral reweighting to input tensor
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
        Returns:
            Reweighted tensor with same shape
        """
        if not self.is_fitted:
            return x  # Return unchanged if PCA not fitted
            
        original_shape = x.shape
        x_flat = x.view(-1, self.embed_dim)
        
        # Center the data
        x_centered = x_flat - self.pca_mean
        
        # Project onto principal components
        pc_projections = torch.matmul(x_centered, self.pca_components)  # [N, n_components]
        
        # Apply learnable reweighting
        weighted_projections = pc_projections * self.pc_weights
        
        # Reconstruct in original space
        reconstructed = torch.matmul(weighted_projections, self.pca_components.T)
        
        # Add back the mean and residual components
        residual = x_centered - torch.matmul(pc_projections, self.pca_components.T)
        reweighted = reconstructed + residual + self.pca_mean
        
        return reweighted.view(original_shape)


class ResiDualHTSAT(HTSAT_Swin_Transformer):
    """
    HTSAT with ResiDual spectral reweighting applied to transformer blocks
    """
    def __init__(self, *args, residual_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.residual_config = residual_config or {
            'n_components_ratio': 0.25,  # Use 25% of embedding dim as components
            'reweight_factor': 2.0,
            'target_layers': [2, 4, 6],  # Which transformer layers to apply reweighting
            'analysis_mode': False  # Set to True for analysis phase
        }
        
        # Add spectral reweighting layers
        self._add_spectral_layers()
        
        # Storage for head analysis
        self.head_outputs = []
        self.attention_maps = []
        
    def _add_spectral_layers(self):
        """Add spectral reweighting layers to target transformer blocks"""
        target_layers = self.residual_config.get('target_layers', [])
        n_components = int(self.embed_dim * self.residual_config.get('n_components_ratio', 0.25))
        
        self.spectral_layers = nn.ModuleDict()
        
        for layer_idx in target_layers:
            if layer_idx < len(self.layers):
                layer_name = f'layer_{layer_idx}'
                self.spectral_layers[layer_name] = SpectralReweightingLayer(
                    embed_dim=int(self.embed_dim * 2 ** layer_idx),
                    n_components=n_components * (2 ** layer_idx),
                    reweight_factor=self.residual_config.get('reweight_factor', 2.0)
                )
    
    def collect_head_outputs(self, layer_idx: int, x: torch.Tensor, attn: torch.Tensor):
        """Collect head outputs for PCA analysis"""
        if self.residual_config.get('analysis_mode', False):
            self.head_outputs.append({
                'layer': layer_idx,
                'output': x.detach().clone(),
                'attention': attn.detach().clone()
            })
    
    def forward_features(self, x):
        """Forward with spectral reweighting"""
        frames_num = x.shape[2] if len(x.shape) > 2 else x.shape[1]
        x = self.patch_embed(x)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # Forward through layers with optional spectral reweighting
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            
            # Collect outputs for analysis
            self.collect_head_outputs(i, x, attn)
            
            # Apply spectral reweighting if configured
            layer_name = f'layer_{i}'
            if layer_name in self.spectral_layers:
                x = self.spectral_layers[layer_name](x)
        
        return super().forward_features(x)  # Continue with rest of forward pass
    
    def fit_spectral_layers(self, dataloader, max_samples: int = 10000):
        """
        Fit PCA components for spectral reweighting layers
        """
        self.eval()
        self.residual_config['analysis_mode'] = True
        
        print("Collecting head outputs for PCA fitting...")
        
        collected_outputs = {layer_name: [] for layer_name in self.spectral_layers.keys()}
        n_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if n_samples >= max_samples:
                    break
                    
                # Forward pass to collect head outputs
                if isinstance(batch, dict):
                    audio = batch.get('audio', batch.get('waveform'))
                else:
                    audio = batch[0]
                
                self.head_outputs = []  # Reset collection
                _ = self.forward(audio)
                
                # Group outputs by layer
                for output_dict in self.head_outputs:
                    layer_idx = output_dict['layer']
                    layer_name = f'layer_{layer_idx}'
                    
                    if layer_name in collected_outputs:
                        output_tensor = output_dict['output']
                        # Flatten spatial dimensions: [B, H*W, C] -> [B*H*W, C]
                        flattened = output_tensor.view(-1, output_tensor.size(-1))
                        collected_outputs[layer_name].append(flattened)
                
                n_samples += audio.size(0)
        
        # Fit PCA for each spectral layer
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


class ResiDualAudioEncoder(AudioEncoder):
    """
    Audio encoder with ResiDual spectral reweighting
    """
    def __init__(self, audioenc_name: str, d_in: int, d_out: int, 
                 sample_rate: int, window_size: int, hop_size: int, 
                 mel_bins: int, fmin: int, fmax: int, classes_num: int,
                 residual_config: Dict = None):
        
        # Initialize base encoder but replace with ResiDual version if HTSAT
        self.residual_config = residual_config or {}
        
        if audioenc_name == "HTSAT":
            # Create ResiDual HTSAT instead of regular HTSAT
            from . import config  # Import HTSAT config
            self.base = ResiDualHTSAT(config=config, residual_config=residual_config)
        else:
            # Use regular encoder for other types
            from .audio import get_audio_encoder
            audio_encoder = get_audio_encoder(audioenc_name)
            self.base = audio_encoder(
                sample_rate, window_size, hop_size, mel_bins, 
                fmin, fmax, classes_num, d_in
            )
        
        # Initialize projection layer
        from .clap import Projection
        self.projection = Projection(d_in, d_out)
    
    def forward(self, x):
        out_dict = self.base(x)
        audio_features = out_dict['embedding']
        audio_classification_output = out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output


class ResiDualCLAP(CLAP):
    """
    CLAP model with ResiDual spectral reweighting
    """
    def __init__(self, audioenc_name: str, sample_rate: int, window_size: int,
                 hop_size: int, mel_bins: int, fmin: int, fmax: int,
                 classes_num: int, out_emb: int, text_model: str,
                 transformer_embed_dim: int, d_proj: int,
                 residual_config: Dict = None):
        
        # Store config before calling parent init
        self.residual_config = residual_config or {}
        
        # Initialize parent without audio encoder
        nn.Module.__init__(self)
        
        # Create ResiDual audio encoder
        self.audio_encoder = ResiDualAudioEncoder(
            audioenc_name, out_emb, d_proj, sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax, classes_num, residual_config
        )
        
        # Initialize text encoder normally
        from .clap import TextEncoder
        self.caption_encoder = TextEncoder(d_proj, text_model, transformer_embed_dim)
        
        # Initialize logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def fit_spectral_components(self, audio_dataloader, max_samples: int = 10000):
        """
        Fit spectral components on audio data
        """
        if hasattr(self.audio_encoder.base, 'fit_spectral_layers'):
            return self.audio_encoder.base.fit_spectral_layers(audio_dataloader, max_samples)
        else:
            print("Audio encoder does not support spectral fitting")
            return {}
