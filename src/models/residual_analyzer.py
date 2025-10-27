import torch
import numpy as np

from .residual_clap import ResiDualCLAP

class ResiDualAnalyzer:
    """
    Analyzer for studying residual stream specialization in audio transformers
    """
    def __init__(self, model: ResiDualCLAP):
        self.model = model
        self.analysis_results = {}
    
    def analyze_head_specialization(self, dataloader, task_labels: List[str] = None):
        """
        Analyze attention head specialization for different audio tasks
        """
        print("Analyzing attention head specialization...")
        
        self.model.eval()
        head_activations = []
        task_correlations = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 100:  # Limit analysis samples
                    break
                
                # Extract audio and process
                audio = batch.get('audio', batch[0])
                
                # Enable analysis mode
                if hasattr(self.model.audio_encoder.base, 'residual_config'):
                    self.model.audio_encoder.base.residual_config['analysis_mode'] = True
                    self.model.audio_encoder.base.head_outputs = []
                
                # Forward pass
                audio_embed, _ = self.model.audio_encoder(audio)
                
                # Collect head outputs
                if hasattr(self.model.audio_encoder.base, 'head_outputs'):
                    for output_dict in self.model.audio_encoder.base.head_outputs:
                        head_activations.append({
                            'layer': output_dict['layer'],
                            'activation': output_dict['output'].mean(dim=1),  # Average over sequence
                            'attention': output_dict['attention'].mean(dim=(1,2))  # Average attention
                        })
        
        # Analyze specialization patterns
        self.analysis_results['head_specialization'] = self._compute_specialization_metrics(head_activations)
        return self.analysis_results['head_specialization']
    
    def _compute_specialization_metrics(self, head_activations):
        """Compute metrics for head specialization"""
        # Group by layer
        layer_activations = {}
        for activation in head_activations:
            layer = activation['layer']
            if layer not in layer_activations:
                layer_activations[layer] = []
            layer_activations[layer].append(activation['activation'])
        
        # Compute variance and correlation patterns
        specialization_metrics = {}
        for layer, activations in layer_activations.items():
            if activations:
                activation_tensor = torch.stack(activations)  # [n_samples, embed_dim]
                
                # Compute variance across samples for each dimension
                variance_per_dim = torch.var(activation_tensor, dim=0)
                
                # Compute correlation matrix
                correlation = torch.corrcoef(activation_tensor.T)
                
                specialization_metrics[f'layer_{layer}'] = {
                    'variance_per_dim': variance_per_dim.cpu().numpy(),
                    'mean_variance': variance_per_dim.mean().item(),
                    'correlation_matrix': correlation.cpu().numpy(),
                    'low_rank_ratio': self._estimate_effective_rank(correlation)
                }
        
        return specialization_metrics
    
    def _estimate_effective_rank(self, correlation_matrix, threshold=0.95):
        """Estimate effective rank of correlation matrix"""
        eigenvals = torch.linalg.eigvals(correlation_matrix).real
        eigenvals_sorted = torch.sort(eigenvals, descending=True)[0]
        cumsum = torch.cumsum(eigenvals_sorted, dim=0)
        total = cumsum[-1]
        
        # Find number of eigenvalues needed to explain threshold of variance
        effective_rank = torch.sum(cumsum / total < threshold).item() + 1
        return effective_rank / len(eigenvals_sorted)
    
    def evaluate_spectral_impact(self, test_dataloader, tasks: List[str]):
        """
        Evaluate impact of spectral reweighting on different audio tasks
        """
        print("Evaluating spectral reweighting impact...")
        
        results = {}
        
        # Test with and without spectral reweighting
        for use_spectral in [False, True]:
            # Enable/disable spectral layers
            for layer_name, spectral_layer in self.model.audio_encoder.base.spectral_layers.items():
                if use_spectral:
                    spectral_layer.pc_weights.data.fill_(2.0)  # Enable reweighting
                else:
                    spectral_layer.pc_weights.data.fill_(1.0)  # Disable reweighting
            
            task_results = {}
            for task in tasks:
                task_results[task] = self._evaluate_task_performance(test_dataloader, task)
            
            results['with_spectral' if use_spectral else 'without_spectral'] = task_results
        
        self.analysis_results['spectral_impact'] = results
        return results
    
    def _evaluate_task_performance(self, dataloader, task: str):
        """Evaluate performance on specific task"""
        # Implement task-specific evaluation
        # This would depend on your specific audio tasks
        
        if task == "audio_classification":
            return self._evaluate_classification(dataloader)
        elif task == "audio_retrieval":
            return self._evaluate_retrieval(dataloader)
        else:
            return {"accuracy": 0.0}  # Placeholder
    
    def _evaluate_classification(self, dataloader):
        """Evaluate audio classification performance"""
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                audio = batch.get('audio', batch[0])
                labels = batch.get('labels', batch[1])
                
                audio_embed, class_logits = self.model.audio_encoder(audio)
                predictions = torch.argmax(class_logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return {"accuracy": correct / total if total > 0 else 0.0}
    
    def _evaluate_retrieval(self, dataloader):
        """Evaluate audio-text retrieval performance"""
        # Placeholder implementation
        return {"recall@1": 0.5, "recall@5": 0.7}