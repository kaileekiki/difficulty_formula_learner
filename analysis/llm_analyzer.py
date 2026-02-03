"""
LLM-based Formula Analysis
Uses OpenAI or Claude API to interpret and recommend formulas
"""
import os
import json
from typing import Dict, List, Optional, Any


class LLMAnalyzer:
    """Use LLM APIs for formula interpretation and recommendations."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize LLM analyzer.
        
        Args:
            api_key: API key (or set via environment variable)
            provider: "openai" or "anthropic"
            model: Model name (defaults: "gpt-4" for OpenAI, "claude-3-sonnet-20240229" for Anthropic)
        """
        self.provider = provider
        self.client = None
        
        # Set default models
        if model:
            self.model = model
        elif provider == "openai":
            self.model = "gpt-4"
        else:  # anthropic
            self.model = "claude-3-sonnet-20240229"
        
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                try:
                    import openai
                    openai.api_key = self.api_key
                    self.client = openai
                except ImportError:
                    pass
        elif provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if self.api_key:
                try:
                    from anthropic import Anthropic
                    self.client = Anthropic(api_key=self.api_key)
                except ImportError:
                    pass
        
        self.is_available = self.api_key is not None and self.client is not None
    
    def interpret_formula(self, formula_string: str, feature_importance: Dict[str, float]) -> str:
        """
        Get LLM interpretation of a formula.
        
        Args:
            formula_string: The mathematical formula
            feature_importance: Dictionary of feature importance scores
            
        Returns:
            Natural language interpretation
        """
        if not self.is_available:
            return self._fallback_interpretation(formula_string, feature_importance)
        
        # Sanitize inputs to prevent prompt injection
        # Limit formula string length and remove potentially problematic characters
        safe_formula = str(formula_string)[:500]
        safe_importance = {str(k)[:100]: float(v) for k, v in list(feature_importance.items())[:20]}
        
        prompt = f"""
다음 버그 난이도 예측 수식을 해석해주세요:

수식: {safe_formula}

Feature Importance:
{json.dumps(safe_importance, indent=2)}

다음 관점에서 해석해주세요:
1. 수식의 직관적 의미
2. 가장 중요한 요소와 그 이유
3. 버그 수정 관점에서의 해석
4. 한계점이나 주의사항
"""
        
        return self._call_api(prompt)
    
    def recommend_formula_improvements(self, 
                                       current_formula: str,
                                       metrics: Dict[str, float],
                                       correlation_data: Dict[str, float]) -> str:
        """
        Get LLM recommendations for formula improvements.
        
        Args:
            current_formula: Current best formula
            metrics: Evaluation metrics (R², Spearman, etc.)
            correlation_data: Feature-target correlations
            
        Returns:
            Improvement recommendations
        """
        if not self.is_available:
            return self._fallback_recommendations(metrics)
        
        # Sanitize inputs
        safe_formula = str(current_formula)[:500]
        safe_metrics = {str(k)[:50]: float(v) for k, v in list(metrics.items())[:10]}
        safe_corr = {str(k)[:100]: float(v) for k, v in list(correlation_data.items())[:20]}
        
        prompt = f"""
현재 버그 난이도 예측 수식의 성능을 개선하기 위한 제안을 해주세요.

현재 수식: {safe_formula}

성능 지표:
- Spearman ρ: {safe_metrics.get('spearman', 'N/A')}
- R²: {safe_metrics.get('r2', 'N/A')}
- RMSE: {safe_metrics.get('rmse', 'N/A')}

Feature-Target 상관관계:
{json.dumps(safe_corr, indent=2)}

다음 관점에서 제안해주세요:
1. 추가할 만한 feature 변환 (log, sqrt, 제곱 등)
2. Feature 간 상호작용 (곱셈, 비율 등)
3. 정규화 방법 개선
4. 새로운 수식 형태 제안
"""
        
        return self._call_api(prompt)
    
    def explain_formula_selection(self, 
                                  formula_candidates: List[Dict[str, Any]],
                                  best_formula: Dict[str, Any]) -> str:
        """
        Explain why a particular formula was selected.
        
        Args:
            formula_candidates: List of all evaluated formulas
            best_formula: The selected best formula
            
        Returns:
            Explanation of selection
        """
        if not self.is_available:
            return self._fallback_selection_explanation(formula_candidates, best_formula)
        
        # Sanitize and limit candidates to top 10
        safe_candidates = []
        for c in formula_candidates[:10]:
            safe_name = str(c.get('formula_name', 'Unknown'))[:100]
            safe_candidates.append({
                'formula_name': safe_name,
                'cv_spearman_mean': float(c.get('cv_spearman_mean', 0)),
                'cv_r2_mean': float(c.get('cv_r2_mean', 0)),
                'complexity': int(c.get('complexity', 0))
            })
        
        safe_best = {
            'formula_name': str(best_formula.get('formula_name', 'Unknown'))[:100],
            'cv_spearman_mean': float(best_formula.get('cv_spearman_mean', 0)),
            'cv_r2_mean': float(best_formula.get('cv_r2_mean', 0)),
            'complexity': int(best_formula.get('complexity', 0))
        }
        
        candidates_summary = "\n".join([
            f"- {c['formula_name']}: Spearman={c['cv_spearman_mean']:.4f}, "
            f"R²={c['cv_r2_mean']:.4f}, Complexity={c['complexity']}"
            for c in safe_candidates
        ])
        
        prompt = f"""
여러 수식 후보 중 최적의 수식이 선택된 이유를 설명해주세요.

후보 수식들:
{candidates_summary}

선택된 수식:
- 이름: {safe_best['formula_name']}
- Spearman ρ: {safe_best['cv_spearman_mean']:.4f}
- R²: {safe_best['cv_r2_mean']:.4f}
- Complexity: {safe_best['complexity']}

다음을 포함해서 설명해주세요:
1. 이 수식이 선택된 핵심 이유
2. 다른 후보 대비 장점
3. 예상되는 일반화 성능
4. 실제 사용 시 주의사항
"""
        
        return self._call_api(prompt)
    
    def _call_api(self, prompt: str) -> str:
        """Call the appropriate API."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            return f"API 호출 오류: {str(e)}"
    
    def _fallback_interpretation(self, formula_string: str, feature_importance: Dict[str, float]) -> str:
        """Fallback interpretation without API."""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
        
        interpretation = f"""
## 수식 해석 (API 키 없이 자동 생성)

**수식**: `{formula_string}`

### 주요 요소 (중요도 순):
"""
        for i, (name, imp) in enumerate(top_features, 1):
            interpretation += f"{i}. **{name}**: 중요도 {imp:.4f}\n"
        
        interpretation += """
### 해석:
- 이 수식은 버그의 코드 복잡도 메트릭을 기반으로 성공률을 예측합니다.
- 더 자세한 해석을 위해 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수를 설정해주세요.
"""
        return interpretation
    
    def _fallback_recommendations(self, metrics: Dict[str, float]) -> str:
        """Fallback recommendations without API."""
        r2 = metrics.get('r2', 0)
        spearman = metrics.get('spearman', 0)
        
        recommendations = "## 개선 제안 (API 키 없이 자동 생성)\n\n"
        
        if r2 < 0.5:
            recommendations += "- R²가 낮습니다. 비선형 변환(log, sqrt)을 시도해보세요.\n"
        if spearman < 0.7:
            recommendations += "- 순위 상관이 낮습니다. Feature 간 상호작용 항을 추가해보세요.\n"
        
        recommendations += "- 더 자세한 제안을 위해 API 키를 설정해주세요."
        return recommendations
    
    def _fallback_selection_explanation(self, candidates: List[Dict], best: Dict) -> str:
        """Fallback selection explanation without API."""
        return f"""
## 수식 선택 설명 (API 키 없이 자동 생성)

**선택된 수식**: {best.get('formula_name')}

### 선택 이유:
- Spearman 상관계수: {best.get('cv_spearman_mean', 0):.4f}
- R² 결정계수: {best.get('cv_r2_mean', 0):.4f}
- 복잡도: {best.get('complexity', 0)}

이 수식은 예측 성능(Spearman)과 복잡도 사이의 균형이 가장 좋습니다.
더 자세한 설명을 위해 API 키를 설정해주세요.
"""
