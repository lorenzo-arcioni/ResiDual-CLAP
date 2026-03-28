# models/residual_clap.py

import torch
import torch.nn as nn
import numpy as np
import random
import math
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from typing import Dict
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
        
        # Inizializza weights basati su explained variance
        variance_ratio = pca.explained_variance_ratio_
        # Esempio: enfatizza componenti con alta variance
        initial_weights = torch.tensor(
            1.0 + self.reweight_factor * variance_ratio,  # Più variance → più peso
            dtype=torch.float32
        )
        self.pc_weights.data = initial_weights
        
        self.is_fitted.data = torch.tensor(True)
        
        return variance_ratio
    
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
    HTSAT con spectral reweighting - EQUIVALENTE ALL'ORIGINALE
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
        """Aggiungi spectral layers con dimensioni reali"""
        target_layers = self.residual_config.get('target_layers', [])
        n_components_ratio = self.residual_config.get('n_components_ratio', 0.25)

        self.spectral_layers = nn.ModuleDict()
        
        # Forward pass di test per trovare dimensioni
        print("🔍 Detecting layer dimensions...")
        self.residual_config['analysis_mode'] = True
        self.head_outputs = []
        
        with torch.no_grad():
            test_audio = torch.randn(1, self.config.clip_samples if hasattr(self.config, 'clip_samples') else 320000)
            if next(self.parameters()).is_cuda:
                test_audio = test_audio.cuda()
            
            try:
                _ = self.forward(test_audio)
            except:
                pass
        for item in self.head_outputs:
            print(f"  ✓ {item['layer']}: {item['output'].shape}")
        # Crea spectral layers
        for layer_idx in target_layers:
            for output_dict in self.head_outputs:
                if output_dict['layer'] == layer_idx:
                    actual_dim = output_dict['output'].shape[-1]
                    layer_n_components = int(actual_dim * n_components_ratio)
                    
                    layer_name = f'layer_{layer_idx}'
                    self.spectral_layers[layer_name] = SpectralReweightingLayer(
                        embed_dim=actual_dim,
                        n_components=layer_n_components,
                        reweight_factor=self.residual_config.get('reweight_factor', 2.0)
                    )
                    print(f"  ✓ {layer_name}: {actual_dim}D → {layer_n_components} PCs")
                    break
        
        self.residual_config['analysis_mode'] = False
        self.head_outputs = []
    
    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False):
        """
        Forward COMPLETO con gestione multi-crop/repeat
        """
        # Preprocessing standard
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
        # ============================================================
        # GESTIONE LUNGHEZZA INPUT (3 modalità)
        # ============================================================
        
        # MODALITÀ A: Infer mode
        if infer_mode:
            frame_num = x.shape[2]
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
            x = self.reshape_wav2img(x)
            output_dict = self._forward_features(x)  # apply_reweighting=True (default)
        
        # MODALITÀ B: Repeat mode
        elif self.config.enable_repeat_mode:
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                output_dict = self._forward_features(x)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    tx = self.repeat_wat2img(tx, cur_pos)
                    output_dicts.append(self._forward_features(tx))
                output_dict = self._average_outputs(output_dicts, x.device)
        
        # MODALITÀ C: Crop mode (default)
        else:
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    x = self.reshape_wav2img(x)
                    output_dict = self._forward_features(x)
                else:
                    overlap_size = 344
                    crop_size = 689
                    output_dicts = []
                    
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
                        tx = self.reshape_wav2img(tx)
                        output_dicts.append(self._forward_features(tx))
                    
                    output_dict = self._average_outputs(output_dicts, x.device, include_latent=True)
            else:
                x = self.reshape_wav2img(x)
                output_dict = self._forward_features(x)
        return output_dict
    
    # ================================================================
    # Core transformer + reweighting (sostituisce forward_features)
    # ================================================================
    def _forward_features(self, x, apply_reweighting=True):
        """
        Forward features con opzione di disabilitare reweighting
        
        Args:
            x: input tensor dopo preprocessing
            apply_reweighting: se False, salta il reweighting (usato per PCA fitting)
        """
        frames_num = x.shape[2]

        # Patch embedding (standard)
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # ============================================================
        # Loop con spectral reweighting opzionale
        # ============================================================
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            
            # 🟢 Salva output per analysis (sempre, indipendentemente dal reweighting)
            if self.residual_config.get('analysis_mode', False):
                self.head_outputs.append({
                    'layer': i,
                    'output': x.detach().clone(),
                    'attention': attn.detach().clone() if attn is not None else None
                })
            
            # 🟢 Applica spectral reweighting solo se abilitato E se apply_reweighting=True
            layer_name = f'layer_{i}'
            if apply_reweighting and layer_name in self.spectral_layers:
                x = self.spectral_layers[layer_name](x)
        # ============================================================
        # RESTO IDENTICO ALL'ORIGINALE forward_features()
        if self.config.enable_tscam:
            # Branch TSCAM
            x = self.norm(x)
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape

            # Group 2D CNN
            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            
            # Latent output
            latent_output = self.avgpool(torch.flatten(x, 2))
            latent_output = torch.flatten(latent_output, 1)

            # Attention heatmap (se abilitato)
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
            
            # TSCAM convolution
            x = self.tscam_conv(x)
            x = torch.flatten(x, 2)  # B, C, T

            # Framewise output
            if self.config.htsat_attn_heatmap:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous() * attn, 8 * self.patch_stride[1])
            else:
                from .pytorch_utils import interpolate
                fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            
            # Clipwise output
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
            # Branch NON-TSCAM
            x = self.norm(x)
            B, N, C = x.shape
            
            # Reshape per framewise
            fpx = x.permute(0, 2, 1).contiguous().reshape(B, C, frames_num // (2 ** (len(self.depths) + 1)), frames_num // (2 ** (len(self.depths) + 1)))
            B, C, F, T = fpx.shape
            
            # Group frequencies
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim=2)
            
            # Interpolate framewise
            from .pytorch_utils import interpolate
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            
            # Clipwise output
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            
            # Applica head
            if self.num_classes > 0:
                x = self.head(x)
                fpx = self.head(fpx)
            
            output_dict = {
                'framewise_output': torch.sigmoid(fpx),
                'clipwise_output': torch.sigmoid(x)
            }
        
        return output_dict
    
    # ================================================================
    # MODIFICA 3: Funzione per mediare output multipli (OK)
    # ================================================================
    def _average_outputs(self, output_dicts, device, include_latent=False):
        """Media i risultati di multiple forward pass"""
        # Inizializza accumulatori
        result = {}
        keys_to_average = ['clipwise_output', 'framewise_output', 'embedding']
        if include_latent:
            keys_to_average.append('latent_output')
        
        # Trova quali chiavi esistono
        available_keys = [k for k in keys_to_average if k in output_dicts[0]]
        
        for key in available_keys:
            accumulated = torch.zeros_like(output_dicts[0][key]).float().to(device)
            for d in output_dicts:
                accumulated += d[key]
            result[key] = accumulated / len(output_dicts)
        
        return result
    
    # ================================================================
    # Fitting
    # ================================================================
    def fit_spectral_layers(self, dataloader, max_samples: int = 10000):
        """Fit PCA per ogni spectral layer"""
        
        self.eval()
        original_analysis_mode = self.residual_config.get('analysis_mode', False)
        self.residual_config['analysis_mode'] = True  # ✅ Abilita collection
        
        print("\n" + "="*80)
        print("🔍 PHASE 1: Collecting Hidden States from HTSAT Layers")
        print("="*80)
        print(f"Target layers: {list(self.spectral_layers.keys())}")
        print(f"Max samples to collect: {max_samples}")
        print(f"Batches in dataloader: {len(dataloader)}")
        
        collected_outputs = {layer_name: [] for layer_name in self.spectral_layers.keys()}
        n_samples = 0
        n_batches = 0
        failed_batches = 0
        
        expected_batches = min(len(dataloader), max_samples // dataloader.batch_size + 1)
        start_time = time.time()
        
        with torch.no_grad():
            pbar = tqdm(
                dataloader, 
                desc="Collecting samples",
                total=expected_batches,
                unit="batch",
                ncols=100
            )
            
            for batch_idx, batch in enumerate(pbar):
                if n_samples >= max_samples:
                    pbar.close()
                    break
            
                # Extract audio
                if isinstance(batch, dict):
                    audio = batch.get('audio', batch.get('waveform'))
                else:
                    audio = batch[0] if isinstance(batch, (list, tuple)) else batch

                batch_size = audio.size(0)
                
                if next(self.parameters()).is_cuda and not audio.is_cuda:
                    audio = audio.cuda()
                
                # Reset buffer
                self.head_outputs = []
                
                try:
                    # ✅ Preprocessing standard
                    x = self.spectrogram_extractor(audio)
                    x = self.logmel_extractor(x)
                    x = x.transpose(1, 3)
                    x = self.bn0(x)
                    x = x.transpose(1, 3)
                    
                    # NO augmentation durante fitting
                    
                    # Reshape standard
                    if x.shape[2] > self.freq_ratio * self.spec_size:
                        x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    
                    x = self.reshape_wav2img(x)

                    # ✅ Forward SENZA reweighting
                    _ = self._forward_features(x, apply_reweighting=False)
                    
                    n_batches += 1
                    
                except Exception as e:
                    failed_batches += 1
                    pbar.set_postfix({
                        'samples': n_samples,
                        'failed': failed_batches,
                        'status': '⚠️ FAILED'
                    })
                    print(e)
                    continue
                
                # Collect outputs
                collected_this_batch = 0
                for output_dict in self.head_outputs:
                    layer_idx = output_dict['layer']
                    layer_name = f'layer_{layer_idx}'
                    
                    if layer_name in collected_outputs:
                        output_tensor = output_dict['output']
                        flattened = output_tensor.view(-1, output_tensor.size(-1))
                        collected_outputs[layer_name].append(flattened.cpu())
                        collected_this_batch += 1
                
                n_samples += batch_size
                
                pbar.set_postfix({
                    'samples': f"{n_samples}/{max_samples}",
                    'layers': f"{collected_this_batch}/{len(self.spectral_layers)}",
                    'failed': failed_batches
                })
            
            pbar.close()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Collection completed in {elapsed_time:.2f}s")
        print(f"  • Total samples processed: {n_samples}")
        print(f"  • Successful batches: {n_batches}")
        print(f"  • Failed batches: {failed_batches}")
        print(f"  • Samples per second: {n_samples/elapsed_time:.1f}")
        
        # Summary of collected data
        print(f"\n📦 Collected data per layer:")
        for layer_name, outputs in collected_outputs.items():
            if outputs:
                total_tokens = sum(o.shape[0] for o in outputs)
                print(f"  • {layer_name}: {len(outputs)} batches, {total_tokens:,} tokens")
            else:
                print(f"  • {layer_name}: ⚠️  NO DATA COLLECTED")
        
        # ========================================================================
        # PHASE 2: Fit PCA
        # ========================================================================
        print("\n" + "="*80)
        print("📊 PHASE 2: Fitting PCA Components")
        print("="*80)
        
        variance_ratios = {}
        
        for layer_name in tqdm(collected_outputs.keys(), desc="Fitting PCA", unit="layer"):
            outputs = collected_outputs[layer_name]
            
            if not outputs:
                print(f"\n⚠️  {layer_name}: Skipping (no data collected)")
                continue
            
            print(f"\n🔧 Processing {layer_name}:")
            
            # Concatenate all batches
            print(f"  • Concatenating {len(outputs)} batches...")
            combined_outputs = torch.cat(outputs, dim=0)
            print(f"  • Combined shape: {combined_outputs.shape} ({combined_outputs.shape[0]:,} samples)")
            
            # Memory info
            memory_mb = combined_outputs.element_size() * combined_outputs.nelement() / (1024**2)
            print(f"  • Memory usage: {memory_mb:.1f} MB")
            
            # Fit PCA
            print(f"  • Running PCA decomposition...")
            pca_start = time.time()
            
            variance_ratio = self.spectral_layers[layer_name].fit_pca(combined_outputs)
            
            pca_time = time.time() - pca_start
            print(f"  • PCA completed in {pca_time:.2f}s")
            
            variance_ratios[layer_name] = variance_ratio
            
            # Detailed variance analysis
            n_components = len(variance_ratio)
            print(f"\n  📈 Variance Analysis:")
            print(f"     • Total components: {n_components}")
            print(f"     • Top 5 variances: {[f'{v:.4f}' for v in variance_ratio[:5]]}")
            print(f"     • Cumulative variance:")
            
            cumsum = np.cumsum(variance_ratio)
            for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
                n_comp = np.searchsorted(cumsum, threshold) + 1
                print(f"        - {threshold*100:.0f}% variance: {n_comp}/{n_components} components ({n_comp/n_components*100:.1f}%)")
            
            # Visualize top components
            print(f"\n     Top 10 components bar:")
            max_var = variance_ratio[0]
            for i in range(min(10, len(variance_ratio))):
                bar_len = int(variance_ratio[i] / max_var * 40)
                bar = "█" * bar_len
                print(f"        PC{i+1:2d}: {bar} {variance_ratio[i]:.4f}")
        
        # ========================================================================
        # Final Summary
        # ========================================================================
        print("\n" + "="*80)
        print("✓ PCA Fitting Completed Successfully")
        print("="*80)
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f}s")
        print(f"\nSpectral layers ready for reweighting:")
        for layer_name in self.spectral_layers.keys():
            is_fitted = self.spectral_layers[layer_name].is_fitted.item()
            status = "✓ FITTED" if is_fitted else "✗ NOT FITTED"
            print(f"  • {layer_name}: {status}")
        
        # Restore original mode
        self.residual_config['analysis_mode'] = original_analysis_mode
        
        return variance_ratios

from . import config

class ResiDualHTSATWrapper(nn.Module):
    def __init__(self, residual_config=None):
        super().__init__()

        # print("parameters are being overidden when using HTSAT")
        # print("HTSAT only support loading a pretrained model on AudioSet")
        # @TODO later look at what parameters are same and can be merged

        self.htsat = ResiDualHTSAT(config=config, residual_config=residual_config)

    def forward(self, x):
        out_dict = self.htsat(x)
        out_dict['embedding'] = out_dict['latent_output']
        return out_dict

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int, residual_config: Dict) -> None:
        super().__init__()

        self.base = ResiDualHTSATWrapper(residual_config=residual_config)

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output
    
# ================================================================
# ResiDualCLAP
# ================================================================

class ResiDualCLAP(CLAP):
    """
    CLAP con ResiDual spectral reweighting
    """
    def __init__(self, *args, residual_config: Dict = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual_config = residual_config or {}

        # sovrascrivo solo l'audio encoder
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
        if hasattr(self.audio_encoder.base.htsat, 'fit_spectral_layers'):
            return self.audio_encoder.base.htsat.fit_spectral_layers(audio_dataloader, max_samples)
        return {}