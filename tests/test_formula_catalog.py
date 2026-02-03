"""
Tests for FormulaCatalog class
"""
import unittest

from formula.formula_catalog import FormulaCatalog, FormulaCandidate


class TestFormulaCatalog(unittest.TestCase):
    """Test FormulaCatalog class."""
    
    def test_get_all_formulas(self):
        """Test getting all formulas."""
        formulas = FormulaCatalog.get_all_formulas()
        self.assertGreater(len(formulas), 0)
        self.assertIsInstance(formulas[0], FormulaCandidate)
    
    def test_get_formulas_by_category(self):
        """Test getting formulas by category."""
        linear_formulas = FormulaCatalog.get_formulas_by_category("linear")
        self.assertGreater(len(linear_formulas), 0)
        
        for f in linear_formulas:
            self.assertEqual(f.category, "linear")
    
    def test_get_formula_by_type(self):
        """Test getting a specific formula by type."""
        formula = FormulaCatalog.get_formula_by_type("linear_ridge")
        self.assertIsNotNone(formula)
        self.assertEqual(formula.formula_type, "linear_ridge")
        self.assertIn("Ridge", formula.name)
    
    def test_get_nonexistent_formula(self):
        """Test getting a nonexistent formula."""
        formula = FormulaCatalog.get_formula_by_type("nonexistent")
        self.assertIsNone(formula)
    
    def test_get_recommended_formulas_small_data(self):
        """Test recommendations for small dataset."""
        recommendations = FormulaCatalog.get_recommended_formulas(
            data_size=50,
            need_interpretability=True,
            has_multicollinearity=False
        )
        
        self.assertGreater(len(recommendations), 0)
        # Should prefer low complexity formulas for small data
        complexities = [f.complexity for f in recommendations[:3]]
        self.assertIn("low", complexities)
    
    def test_get_recommended_formulas_large_data(self):
        """Test recommendations for large dataset."""
        recommendations = FormulaCatalog.get_recommended_formulas(
            data_size=1000,
            need_interpretability=False,
            has_multicollinearity=True
        )
        
        self.assertGreater(len(recommendations), 0)
    
    def test_generate_catalog_report(self):
        """Test generating catalog report."""
        report = FormulaCatalog.generate_catalog_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("수식 후보 카탈로그", report)
        self.assertIn("선형 수식", report)
        self.assertIn("Ridge", report)
        self.assertIn("장점", report)
        self.assertIn("단점", report)
    
    def test_formula_candidate_fields(self):
        """Test that formula candidates have all required fields."""
        formulas = FormulaCatalog.get_all_formulas()
        
        for f in formulas:
            self.assertIsNotNone(f.name)
            self.assertIsNotNone(f.category)
            self.assertIsNotNone(f.formula_type)
            self.assertIsNotNone(f.description)
            self.assertIsNotNone(f.rationale)
            self.assertIsInstance(f.advantages, list)
            self.assertIsInstance(f.disadvantages, list)
            self.assertIsNotNone(f.best_for)
            self.assertIn(f.complexity, ["low", "medium", "high"])
            self.assertIn(f.interpretability, ["low", "medium", "high"])
    
    def test_categories_exist(self):
        """Test that all expected categories exist."""
        expected_categories = ["linear", "polynomial", "nonlinear", "tree", "symbolic"]
        
        for cat in expected_categories:
            formulas = FormulaCatalog.get_formulas_by_category(cat)
            self.assertGreater(len(formulas), 0, f"No formulas found for category: {cat}")


if __name__ == '__main__':
    unittest.main()
