"""
Importance Plots
Visualize feature importance
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os


class ImportancePlotter:
    """Create feature importance visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (10, 6), dpi: int = 300):
        """
        Initialize importance plotter.
        
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
            pass  # Use default if style not available
    
    def plot_importance_bar(self,
                          importance: Dict[str, float],
                          title: str = "Feature Importance",
                          output_path: Optional[str] = None,
                          horizontal: bool = True) -> None:
        """
        Plot feature importance as bar chart.
        
        Args:
            importance: Dictionary mapping feature names to importance scores
            title: Plot title
            output_path: Path to save figure (if None, displays only)
            horizontal: If True, horizontal bars; if False, vertical bars
        """
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if horizontal:
            ax.barh(features, values, color='steelblue')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            ax.invert_yaxis()  # Highest at top
        else:
            ax.bar(features, values, color='steelblue')
            ax.set_ylabel('Importance Score')
            ax.set_xlabel('Feature')
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved importance plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_importance_comparison(self,
                                  importance_dict: Dict[str, Dict[str, float]],
                                  title: str = "Feature Importance Comparison",
                                  output_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple importance methods.
        
        Args:
            importance_dict: Dictionary mapping method names to importance dictionaries
            title: Plot title
            output_path: Path to save figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(importance_dict).fillna(0)
        
        # Sort by average importance
        df['average'] = df.mean(axis=1)
        df = df.sort_values('average', ascending=False)
        df = df.drop('average', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot grouped bars
        df.plot(kind='barh', ax=ax)
        
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comparison plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_importance_heatmap(self,
                               importance_dict: Dict[str, Dict[str, float]],
                               title: str = "Feature Importance Heatmap",
                               output_path: Optional[str] = None) -> None:
        """
        Plot importance as heatmap across methods.
        
        Args:
            importance_dict: Dictionary mapping method names to importance dictionaries
            title: Plot title
            output_path: Path to save figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(importance_dict).fillna(0)
        
        # Sort by average importance
        df['average'] = df.mean(axis=1)
        df = df.sort_values('average', ascending=False)
        df = df.drop('average', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), max(6, len(df) * 0.4)), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Importance'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Method')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved heatmap to {output_path}")
        else:
            plt.show()
        
        plt.close()
