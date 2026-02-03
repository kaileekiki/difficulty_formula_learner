"""
Prediction Plots
Visualize prediction quality and errors
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


class PredictionPlotter:
    """Create prediction visualization plots."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (10, 6), dpi: int = 300):
        """
        Initialize prediction plotter.
        
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
    
    def plot_actual_vs_predicted(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                title: str = "Actual vs Predicted Success Rate",
                                output_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values with regression line.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'b-', linewidth=2, alpha=0.7, label=f'Fit (R²={r_value**2:.3f})')
        
        ax.set_xlabel('Actual Success Rate', fontsize=12)
        ax.set_ylabel('Predicted Success Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved actual vs predicted plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      title: str = "Residual Distribution",
                      output_path: Optional[str] = None) -> None:
        """
        Plot residual distribution histogram.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            output_path: Path to save figure
        """
        residuals = y_pred - y_true
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Histogram
        ax.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add normal distribution curve
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30
        ax.plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved residuals plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_by_bug(self,
                         bug_ids: list,
                         errors: np.ndarray,
                         top_n: int = 20,
                         title: str = "Top Prediction Errors by Bug",
                         output_path: Optional[str] = None) -> None:
        """
        Plot prediction errors for individual bugs.
        
        Args:
            bug_ids: List of bug identifiers
            errors: Array of prediction errors
            top_n: Number of top errors to show
            title: Plot title
            output_path: Path to save figure
        """
        # Get top N errors by absolute value
        abs_errors = np.abs(errors)
        top_indices = np.argsort(abs_errors)[-top_n:][::-1]
        
        top_bugs = [bug_ids[i] for i in top_indices]
        top_errors = errors[top_indices]
        
        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.5), 8), dpi=self.dpi)
        
        # Color by sign
        colors = ['red' if e > 0 else 'blue' for e in top_errors]
        
        ax.barh(range(len(top_bugs)), top_errors, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_bugs)))
        ax.set_yticklabels(top_bugs, fontsize=9)
        ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        ax.set_ylabel('Bug ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Over-prediction'),
            Patch(facecolor='blue', alpha=0.7, label='Under-prediction')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved error by bug plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_prediction_comparison(self,
                                  y_true: pd.Series,
                                  predictions: dict,
                                  title: str = "Prediction Comparison",
                                  output_path: Optional[str] = None) -> None:
        """
        Compare predictions from multiple approaches.
        
        Args:
            y_true: True values
            predictions: Dictionary mapping approach names to predicted values
            title: Plot title
            output_path: Path to save figure
        """
        n_approaches = len(predictions)
        fig, axes = plt.subplots(1, n_approaches, figsize=(n_approaches * 6, 6), dpi=self.dpi)
        
        if n_approaches == 1:
            axes = [axes]
        
        for ax, (name, y_pred) in zip(axes, predictions.items()):
            # Align indices
            common_idx = y_true.index.intersection(y_pred.index)
            y_t = y_true.loc[common_idx].values
            y_p = y_pred.loc[common_idx].values
            
            # Scatter plot
            ax.scatter(y_t, y_p, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(y_t.min(), y_p.min())
            max_val = max(y_t.max(), y_p.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Compute R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_t, y_p)
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}\n(R²={r2:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comparison plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
