"""
Symbolic Regression
Automatically discover interpretable formulas using genetic programming
"""
import pandas as pd
import numpy as np
try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False
from typing import Dict

from .base_formula import BaseFormula


class SymbolicRegressionFormula(BaseFormula):
    """Symbolic regression using genetic programming."""
    
    def __init__(self,
                 population_size: int = 5000,
                 generations: int = 20,
                 tournament_size: int = 20,
                 const_range: tuple = (-1.0, 1.0),
                 init_depth: tuple = (2, 6),
                 function_set: tuple = ('add', 'sub', 'mul', 'div', 'sqrt', 'log'),
                 metric: str = 'mean absolute error',
                 parsimony_coefficient: float = 0.001,
                 random_state: int = 42):
        """
        Initialize symbolic regression formula.
        
        Args:
            population_size: Number of programs in population
            generations: Number of generations to evolve
            tournament_size: Size of tournament for selection
            const_range: Range of constants
            init_depth: Range of initial tree depth
            function_set: Mathematical functions to use
            metric: Fitness metric
            parsimony_coefficient: Penalty for complexity
            random_state: Random seed
        """
        super().__init__(name="SymbolicRegression")
        
        if not HAS_GPLEARN:
            raise ImportError("gplearn is not installed. Install with: pip install gplearn")
        
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.const_range = const_range
        self.init_depth = init_depth
        self.function_set = function_set
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        
        self.model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            const_range=const_range,
            init_depth=init_depth,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            random_state=random_state,
            verbose=0
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SymbolicRegressionFormula':
        """Fit the symbolic regression model."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the evolved formula."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_formula_string(self) -> str:
        """Get the evolved formula as string."""
        if not self.is_fitted:
            return "Not fitted"
        
        # Convert program to string with feature names
        formula_str = str(self.model._program)
        
        # Replace X0, X1, etc. with actual feature names
        for i, name in enumerate(self.feature_names):
            formula_str = formula_str.replace(f'X{i}', name)
        
        return f"success_rate = {formula_str}"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on frequency in evolved formula."""
        if not self.is_fitted:
            return {}
        
        formula_str = str(self.model._program)
        importance = {}
        
        for i, name in enumerate(self.feature_names):
            # Count occurrences of Xi in the formula
            count = formula_str.count(f'X{i}')
            importance[name] = float(count)
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def get_complexity(self) -> int:
        """Get complexity as tree depth."""
        if not self.is_fitted:
            return 0
        return self.model._program.depth()
    
    def get_fitness(self) -> float:
        """Get fitness score of the evolved formula."""
        if not self.is_fitted:
            return None
        return float(self.model._program.fitness_)
