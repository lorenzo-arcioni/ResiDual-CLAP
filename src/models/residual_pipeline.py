# Usage example and training pipeline
from typing import Dict, List

from .residual_clap import ResiDualCLAP
from .residual_analyzer import ResiDualAnalyzer

class ResiDualTrainingPipeline:
    """
    Complete training pipeline for ResiDual audio models
    """
    def __init__(self, model_config: Dict, residual_config: Dict):
        self.model_config = model_config
        self.residual_config = residual_config
        self.model = None
        self.analyzer = None
    
    def initialize_model(self):
        """Initialize ResiDual CLAP model"""
        self.model = ResiDualCLAP(**self.model_config, residual_config=self.residual_config)
        self.analyzer = ResiDualAnalyzer(self.model)
    
    def run_analysis_phase(self, audio_dataloader, text_queries: List[str] = None):
        """
        Phase 1: Analyze model and fit spectral components
        """
        print("=== Phase 1: Analysis and PCA Fitting ===")
        
        # Fit spectral components
        variance_ratios = self.model.fit_spectral_components(audio_dataloader)
        
        # Analyze head specialization
        specialization = self.analyzer.analyze_head_specialization(audio_dataloader, text_queries)
        
        return {
            'variance_ratios': variance_ratios,
            'specialization': specialization
        }
    
    def run_evaluation_phase(self, test_dataloader, tasks: List[str]):
        """
        Phase 2: Evaluate spectral reweighting impact
        """
        print("=== Phase 2: Evaluation ===")
        
        return self.analyzer.evaluate_spectral_impact(test_dataloader, tasks)
    
    def save_analysis_results(self, save_path: str):
        """Save analysis results"""
        import pickle
        
        results = {
            'model_config': self.model_config,
            'residual_config': self.residual_config,
            'analysis_results': self.analyzer.analysis_results
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {save_path}")