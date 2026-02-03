"""
Formula Catalog
Comprehensive list of formula candidates with explanations
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class FormulaCandidate:
    """Represents a formula candidate with explanation."""
    name: str
    category: str
    formula_type: str
    description: str
    rationale: str
    advantages: List[str]
    disadvantages: List[str]
    best_for: str
    complexity: str  # "low", "medium", "high"
    interpretability: str  # "high", "medium", "low"


class FormulaCatalog:
    """Catalog of all available formula types with detailed explanations."""
    
    FORMULAS = [
        # ==================== LINEAR FORMULAS ====================
        FormulaCandidate(
            name="Linear Regression (No Regularization)",
            category="linear",
            formula_type="linear_none",
            description="기본 선형 회귀: success_rate = w₁x₁ + w₂x₂ + ... + b",
            rationale="""
**왜 적합한가?**
- 가장 단순하고 해석하기 쉬운 모델
- 각 메트릭이 성공률에 미치는 영향을 직접적으로 확인 가능
- 계수(w)의 부호로 양/음의 영향 파악 가능

**버그 난이도 예측에서의 의미:**
- 양수 계수: 해당 메트릭이 높을수록 성공률 증가 (쉬운 버그)
- 음수 계수: 해당 메트릭이 높을수록 성공률 감소 (어려운 버그)
""",
            advantages=[
                "완벽한 해석 가능성",
                "계산 효율적",
                "과적합 위험 낮음 (적은 데이터에서도 안정적)",
                "계수로 feature importance 직접 확인"
            ],
            disadvantages=[
                "비선형 관계 포착 불가",
                "Feature 간 상호작용 무시",
                "예측 정확도가 낮을 수 있음"
            ],
            best_for="빠른 기준선(baseline) 모델, 해석 가능성이 중요할 때",
            complexity="low",
            interpretability="high"
        ),
        
        FormulaCandidate(
            name="Ridge Regression (L2)",
            category="linear",
            formula_type="linear_ridge",
            description="L2 정규화 선형 회귀: 계수 크기에 패널티 부여",
            rationale="""
**왜 적합한가?**
- 다중공선성(multicollinearity) 문제 해결
- 13개 메트릭 중 상관관계 높은 것들(예: DFG_GED와 PDG_GED)이 있을 때 유용
- 계수가 극단적으로 커지는 것을 방지

**버그 난이도 예측에서의 의미:**
- Graph 기반 메트릭들(CFG, DFG, PDG, CPG)은 서로 상관관계가 높음
- Ridge가 이들의 계수를 적절히 조절하여 안정적인 예측
""",
            advantages=[
                "다중공선성에 강건",
                "해석 가능성 유지",
                "일반화 성능 향상",
                "모든 feature 유지 (희소성 없음)"
            ],
            disadvantages=[
                "Feature selection 기능 없음",
                "alpha 하이퍼파라미터 튜닝 필요"
            ],
            best_for="Feature 간 상관관계가 높을 때, 모든 메트릭을 유지하고 싶을 때",
            complexity="low",
            interpretability="high"
        ),
        
        FormulaCandidate(
            name="Lasso Regression (L1)",
            category="linear",
            formula_type="linear_lasso",
            description="L1 정규화 선형 회귀: 불필요한 feature의 계수를 0으로",
            rationale="""
**왜 적합한가?**
- 자동 feature selection 기능
- 13개 메트릭 중 실제로 중요한 것만 선택
- 희소 모델(sparse model) 생성으로 해석 용이

**버그 난이도 예측에서의 의미:**
- 13개 메트릭 중 어떤 것이 실제로 성공률 예측에 중요한지 파악
- 예: DFG_GED만 중요하고 나머지는 노이즈라면, Lasso가 이를 자동 발견
""",
            advantages=[
                "자동 feature selection",
                "해석 가능한 희소 모델",
                "불필요한 메트릭 제거"
            ],
            disadvantages=[
                "상관관계 높은 feature 중 하나만 선택 (정보 손실)",
                "alpha 튜닝 민감"
            ],
            best_for="중요한 메트릭만 찾고 싶을 때, 단순한 수식이 필요할 때",
            complexity="low",
            interpretability="high"
        ),
        
        FormulaCandidate(
            name="ElasticNet (L1 + L2)",
            category="linear",
            formula_type="linear_elasticnet",
            description="L1과 L2 정규화를 결합: Ridge와 Lasso의 장점 결합",
            rationale="""
**왜 적합한가?**
- Ridge의 안정성 + Lasso의 feature selection
- 상관관계 높은 feature들을 그룹으로 처리
- Graph 메트릭들처럼 유사한 feature 그룹에 효과적

**버그 난이도 예측에서의 의미:**
- Graph 메트릭 그룹(DFG, PDG, CFG, CPG)을 함께 선택하거나 제외
- Basic 메트릭과 Graph 메트릭의 상대적 중요도 파악
""",
            advantages=[
                "Ridge와 Lasso의 장점 결합",
                "Feature 그룹 처리 가능",
                "유연한 정규화"
            ],
            disadvantages=[
                "두 개의 하이퍼파라미터 (alpha, l1_ratio)",
                "튜닝 복잡"
            ],
            best_for="Feature 그룹이 있을 때, 균형 잡힌 모델이 필요할 때",
            complexity="low",
            interpretability="high"
        ),
        
        # ==================== POLYNOMIAL FORMULAS ====================
        FormulaCandidate(
            name="Polynomial Regression (Degree 2)",
            category="polynomial",
            formula_type="polynomial_2",
            description="2차 다항식: x², x₁×x₂ 항 포함",
            rationale="""
**왜 적합한가?**
- 비선형 관계 포착 (예: 성공률이 메트릭의 제곱에 반비례)
- Feature 간 상호작용 모델링
- 예: LOC×DFG_GED 상호작용 (코드 크기와 데이터 흐름 복잡도의 조합)

**버그 난이도 예측에서의 의미:**
- 단순히 DFG_GED가 높으면 어려운 것이 아니라,
- LOC가 크면서 DFG_GED도 높을 때 특히 어려울 수 있음 (상호작용 효과)
""",
            advantages=[
                "비선형 관계 포착",
                "Feature 상호작용 모델링",
                "선형보다 높은 예측력"
            ],
            disadvantages=[
                "Feature 수 폭발 (13개 → 104개)",
                "과적합 위험",
                "해석 복잡"
            ],
            best_for="선형 모델로 부족할 때, 상호작용이 의심될 때",
            complexity="medium",
            interpretability="medium"
        ),
        
        FormulaCandidate(
            name="Polynomial Regression (Degree 3)",
            category="polynomial",
            formula_type="polynomial_3",
            description="3차 다항식: x³, x²y, xyz 항 포함",
            rationale="""
**왜 적합한가?**
- 더 복잡한 비선형 관계 포착
- 극단적인 값에서의 행동 모델링
- 예: 매우 높은 복잡도에서 성공률이 급격히 떨어지는 패턴

**버그 난이도 예측에서의 의미:**
- 복잡도 메트릭이 특정 임계값을 넘으면 난이도가 급증하는 패턴
- S-curve 형태의 난이도-성공률 관계
""",
            advantages=[
                "복잡한 비선형 패턴 포착",
                "극단값 행동 모델링"
            ],
            disadvantages=[
                "매우 높은 과적합 위험",
                "Feature 수 매우 많음",
                "해석 거의 불가능"
            ],
            best_for="데이터가 충분하고 복잡한 패턴이 있을 때 (주의 필요)",
            complexity="high",
            interpretability="low"
        ),
        
        # ==================== NONLINEAR FORMULAS ====================
        FormulaCandidate(
            name="Log-Transformed Linear",
            category="nonlinear",
            formula_type="nonlinear_log",
            description="log(1 + x) 변환 후 선형 회귀",
            rationale="""
**왜 적합한가?**
- 왜도(skewness)가 높은 메트릭 정규화
- 큰 값의 영향 완화 (예: LOC가 매우 큰 파일)
- 로그 척도에서 선형 관계 (지수적 관계 모델링)

**버그 난이도 예측에서의 의미:**
- LOC가 100→200 증가와 1000→1100 증가의 영향이 다름
- log 변환으로 비율적 변화를 선형화
- "10배 더 복잡하면 X% 더 어려움" 형태의 관계
""",
            advantages=[
                "왜도 높은 데이터 처리",
                "이상치 영향 완화",
                "해석 가능 (로그 척도)"
            ],
            disadvantages=[
                "0 또는 음수 값 처리 필요",
                "로그 척도 해석 필요"
            ],
            best_for="메트릭 분포가 왜곡되어 있을 때, 큰 값의 영향을 줄이고 싶을 때",
            complexity="low",
            interpretability="medium"
        ),
        
        FormulaCandidate(
            name="Sqrt-Transformed Linear",
            category="nonlinear",
            formula_type="nonlinear_sqrt",
            description="sqrt(x) 변환 후 선형 회귀",
            rationale="""
**왜 적합한가?**
- log보다 완만한 변환
- 카운트 데이터에 적합 (예: LOC, Token_Edit_Distance)
- Poisson 분포 가정 데이터에 효과적

**버그 난이도 예측에서의 의미:**
- "복잡도가 4배가 되면 난이도는 2배" 형태의 관계
- LOC가 100→400이면 sqrt 관점에서 10→20으로 선형적
""",
            advantages=[
                "log보다 완만한 변환",
                "카운트 데이터에 적합",
                "음수 없음"
            ],
            disadvantages=[
                "비선형 정도가 제한적"
            ],
            best_for="카운트 기반 메트릭(LOC, Token 등)에 적합",
            complexity="low",
            interpretability="medium"
        ),
        
        # ==================== TREE-BASED FORMULAS ====================
        FormulaCandidate(
            name="Random Forest",
            category="tree",
            formula_type="tree_rf",
            description="여러 결정 트리의 앙상블",
            rationale="""
**왜 적합한가?**
- 자동으로 비선형 관계와 상호작용 학습
- Feature importance 제공
- 과적합에 강건

**버그 난이도 예측에서의 의미:**
- "DFG_GED > 10이고 LOC > 500이면 어려움" 같은 규칙 자동 학습
- 13개 메트릭의 복잡한 조합 패턴 포착
- Permutation importance로 중요 메트릭 파악
""",
            advantages=[
                "비선형/상호작용 자동 학습",
                "과적합에 강건",
                "Feature importance 제공",
                "하이퍼파라미터 민감도 낮음"
            ],
            disadvantages=[
                "해석 어려움 (블랙박스)",
                "수식으로 표현 불가",
                "모델 크기가 큼"
            ],
            best_for="예측 정확도가 최우선일 때, feature importance만 필요할 때",
            complexity="high",
            interpretability="low"
        ),
        
        FormulaCandidate(
            name="XGBoost",
            category="tree",
            formula_type="tree_xgb",
            description="Gradient Boosting 기반 트리 앙상블",
            rationale="""
**왜 적합한가?**
- 일반적으로 Random Forest보다 높은 정확도
- 정규화 내장으로 과적합 방지
- SHAP 값으로 개별 예측 설명 가능

**버그 난이도 예측에서의 의미:**
- 가장 정확한 예측 모델
- SHAP으로 "이 버그가 왜 어려운지" 개별 설명 가능
""",
            advantages=[
                "최고 수준의 예측 정확도",
                "SHAP 기반 해석 가능",
                "정규화 내장"
            ],
            disadvantages=[
                "수식으로 표현 불가",
                "많은 하이퍼파라미터",
                "학습 시간 길 수 있음"
            ],
            best_for="최고의 예측 정확도가 필요할 때",
            complexity="high",
            interpretability="low"
        ),
        
        # ==================== SYMBOLIC REGRESSION ====================
        FormulaCandidate(
            name="Symbolic Regression (gplearn)",
            category="symbolic",
            formula_type="symbolic_gp",
            description="유전 프로그래밍으로 수식 자동 발견",
            rationale="""
**왜 적합한가?**
- 인간이 읽을 수 있는 수식 자동 생성
- 선형, 비선형, 상호작용 모두 탐색
- 복잡도와 정확도 균형 자동 조절 (parsimony)

**버그 난이도 예측에서의 의미:**
- "success_rate = 0.8 - 0.1*sqrt(DFG_GED) - 0.05*log(LOC)" 같은 수식 자동 발견
- 도메인 전문가가 검토하고 의미 부여 가능
- 논문이나 문서에 직접 사용 가능한 수식
""",
            advantages=[
                "해석 가능한 수식 생성",
                "다양한 함수 조합 탐색",
                "복잡도 제어 가능",
                "새로운 인사이트 제공 가능"
            ],
            disadvantages=[
                "계산 비용 높음",
                "결과 재현성 낮을 수 있음",
                "최적 수식 보장 안 됨"
            ],
            best_for="해석 가능한 수식이 필요할 때, 새로운 관계 발견이 목표일 때",
            complexity="medium",
            interpretability="high"
        ),
    ]
    
    @classmethod
    def get_all_formulas(cls) -> List[FormulaCandidate]:
        """Get all formula candidates."""
        return cls.FORMULAS
    
    @classmethod
    def get_formulas_by_category(cls, category: str) -> List[FormulaCandidate]:
        """Get formulas by category."""
        return [f for f in cls.FORMULAS if f.category == category]
    
    @classmethod
    def get_formula_by_type(cls, formula_type: str) -> Optional[FormulaCandidate]:
        """Get a specific formula by type."""
        for f in cls.FORMULAS:
            if f.formula_type == formula_type:
                return f
        return None
    
    @classmethod
    def get_recommended_formulas(cls, 
                                 data_size: int,
                                 need_interpretability: bool = True,
                                 has_multicollinearity: bool = False) -> List[FormulaCandidate]:
        """
        Get recommended formulas based on requirements.
        
        Args:
            data_size: Number of data points
            need_interpretability: Whether interpretability is important
            has_multicollinearity: Whether features are correlated
        """
        recommendations = []
        
        for f in cls.FORMULAS:
            score = 0
            
            # Data size consideration
            if data_size < 100:
                if f.complexity == "low":
                    score += 2
                elif f.complexity == "medium":
                    score += 1
            else:
                score += 1  # All formulas viable with enough data
            
            # Interpretability
            if need_interpretability:
                if f.interpretability == "high":
                    score += 3
                elif f.interpretability == "medium":
                    score += 1
            
            # Multicollinearity
            if has_multicollinearity:
                if f.formula_type in ['linear_ridge', 'linear_elasticnet', 'tree_rf', 'tree_xgb']:
                    score += 2
            
            if score >= 3:
                recommendations.append((f, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in recommendations]
    
    @classmethod
    def generate_catalog_report(cls) -> str:
        """Generate a markdown report of all formulas."""
        report = "# 수식 후보 카탈로그\n\n"
        
        categories = {
            'linear': '선형 수식 (Linear)',
            'polynomial': '다항식 수식 (Polynomial)',
            'nonlinear': '비선형 변환 (Nonlinear)',
            'tree': '트리 기반 (Tree-based)',
            'symbolic': '기호적 회귀 (Symbolic Regression)'
        }
        
        for cat, cat_name in categories.items():
            formulas = cls.get_formulas_by_category(cat)
            if not formulas:
                continue
            
            report += f"## {cat_name}\n\n"
            
            for f in formulas:
                report += f"### {f.name}\n\n"
                report += f"**설명**: {f.description}\n\n"
                report += f"{f.rationale}\n\n"
                
                report += "**장점**:\n"
                for adv in f.advantages:
                    report += f"- {adv}\n"
                report += "\n"
                
                report += "**단점**:\n"
                for dis in f.disadvantages:
                    report += f"- {dis}\n"
                report += "\n"
                
                report += f"**적합한 상황**: {f.best_for}\n\n"
                report += f"**복잡도**: {f.complexity} | **해석 가능성**: {f.interpretability}\n\n"
                report += "---\n\n"
        
        return report
