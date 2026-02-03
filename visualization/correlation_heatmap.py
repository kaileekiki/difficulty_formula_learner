"""
Correlation Heatmap
Visualize correlations between metrics and with target
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import spearmanr


class CorrelationHeatmapPlotter:
    """Create correlation heatmap visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (12, 10), dpi: int = 300):
        """
        Initialize correlation heatmap plotter.
        
        Args:
            style: Matplotlib style
            figsize: Figure size (width, height)
            dpi: Figure DPI
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        try:
            plt.style.use(style)
        except:
            pass
    
    def plot_feature_correlation(self,
                                X: pd.DataFrame,
                                method: str = 'spearman',
                                title: str = "Feature Correlation Matrix",
                                output_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix for features.
        
        Args:
            X: Feature matrix
            method: Correlation method ('spearman', 'pearson', 'kendall')
            title: Plot title
            output_path: Path to save figure
        """
        # Compute correlation matrix
        if method == 'spearman':
            corr_matrix = X.corr(method='spearman')
        elif method == 'pearson':
            corr_matrix = X.corr(method='pearson')
        elif method == 'kendall':
            corr_matrix = X.corr(method='kendall')
        else:
            corr_matrix = X.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': f'{method.capitalize()} Correlation'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved correlation heatmap to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_target_correlation(self,
                               X: pd.DataFrame,
                               y: pd.Series,
                               method: str = 'spearman',
                               title: str = "Feature-Target Correlation",
                               output_path: Optional[str] = None) -> None:
        """
        Plot correlation of each feature with target variable.
        
        Args:
            X: Feature matrix
            y: Target values
            method: Correlation method
            title: Plot title
            output_path: Path to save figure
        """
        # Compute correlations
        correlations = {}
        
        for col in X.columns:
            if method == 'spearman':
                corr, _ = spearmanr(X[col], y)
            elif method == 'pearson':
                from scipy.stats import pearsonr
                corr, _ = pearsonr(X[col], y)
            elif method == 'kendall':
                from scipy.stats import kendalltau
                corr, _ = kendalltau(X[col], y)
            else:
                corr = X[col].corr(y)
            
            correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.5), 6), dpi=self.dpi)
        
        # Color bars by sign
        colors = ['red' if v < 0 else 'green' for v in values]
        
        ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel(f'{method.capitalize()} Correlation with Success Rate')
        ax.set_ylabel('Feature')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (feature, value) in enumerate(zip(features, values)):
            ax.text(value + 0.01 if value >= 0 else value - 0.01, i, f'{value:.3f}',
                   va='center', ha='left' if value >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved target correlation plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_by_tier(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                tiers: dict,
                                method: str = 'spearman',
                                title: str = "Feature-Target Correlation by Tier",
                                output_path: Optional[str] = None) -> None:
        """
        Plot correlations grouped by metric tiers.
        
        Args:
            X: Feature matrix
            y: Target values
            tiers: Dictionary mapping tier names to lists of feature names
            method: Correlation method
            title: Plot title
            output_path: Path to save figure
        """
        # Compute correlations for each tier
        tier_data = {}
        
        for tier_name, features in tiers.items():
            tier_corr = {}
            for feature in features:
                if feature in X.columns:
                    if method == 'spearman':
                        corr, _ = spearmanr(X[feature], y)
                    elif method == 'pearson':
                        from scipy.stats import pearsonr
                        corr, _ = pearsonr(X[feature], y)
                    else:
                        corr = X[feature].corr(y)
                    tier_corr[feature] = corr
            tier_data[tier_name] = tier_corr
        
        # Create subplots for each tier
        n_tiers = len(tier_data)
        fig, axes = plt.subplots(1, n_tiers, figsize=(n_tiers * 5, 6), dpi=self.dpi)
        
        if n_tiers == 1:
            axes = [axes]
        
        for ax, (tier_name, correlations) in zip(axes, tier_data.items()):
            if not correlations:
                continue
            
            sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            features = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            colors = ['red' if v < 0 else 'green' for v in values]
            
            ax.barh(features, values, color=colors, alpha=0.7)
            ax.set_xlabel(f'{method.capitalize()} Correlation')
            ax.set_title(tier_name, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved tier correlation plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
