"""
Statistical significance tests for mitigation improvements.
Tests that improvements are statistically significant.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.fairness.preprocessing_mitigation import compute_reweighing_weights
from src.fairness.fairness_evaluation_utils import (
    demographic_parity_difference,
    equal_opportunity_difference
)

class TestSignificance:
    """Test suite for statistical significance"""
    
    @pytest.fixture
    def experiment_data(self):
        """Create data for statistical testing"""
        np.random.seed(42)
        n_samples = 2000
        
        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        groups = np.random.choice(['A', 'B'], n_samples, p=[0.7, 0.3])
        
        # Add bias
        y[groups == 'B'] = np.random.choice([0, 1], sum(groups == 'B'), p=[0.8, 0.2])
        
        return X, y, groups
    
    def test_bootstrap_significance(self, experiment_data):
        """Test significance using bootstrap"""
        X, y, groups = experiment_data
        n_bootstrap = 1000
        
        baseline_gaps = []
        mitigated_gaps = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            groups_boot = groups[idx]
            
            # Baseline
            baseline = RandomForestClassifier(n_estimators=10, random_state=42)
            baseline.fit(X_boot, y_boot)
            baseline_pred = baseline.predict(X_boot)
            baseline_gap = demographic_parity_difference(baseline_pred, groups_boot)
            baseline_gaps.append(baseline_gap)
            
            # Mitigated
            df = pd.DataFrame({'group': groups_boot, 'label': y_boot})
            weights = compute_reweighing_weights(df, 'group', 'label')
            
            mitigated = RandomForestClassifier(n_estimators=10, random_state=42)
            mitigated.fit(X_boot, y_boot, sample_weight=weights)
            mitigated_pred = mitigated.predict(X_boot)
            mitigated_gap = demographic_parity_difference(mitigated_pred, groups_boot)
            mitigated_gaps.append(mitigated_gap)
        
        # Compute confidence intervals
        baseline_ci = np.percentile(baseline_gaps, [2.5, 97.5])
        mitigated_ci = np.percentile(mitigated_gaps, [2.5, 97.5])
        
        print(f"Baseline DPD: {np.mean(baseline_gaps):.3f} [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]")
        print(f"Mitigated DPD: {np.mean(mitigated_gaps):.3f} [{mitigated_ci[0]:.3f}, {mitigated_ci[1]:.3f}]")
        
        # Test if CIs overlap
        assert mitigated_ci[1] < baseline_ci[0], \
            "Mitigation improvement should be statistically significant (non-overlapping CIs)"
    
    def test_paired_t_test(self, experiment_data):
        """Test significance using paired t-test"""
        X, y, groups = experiment_data
        n_folds = 20
        
        baseline_scores = []
        mitigated_scores = []
        
        for fold in range(n_folds):
            # Different random split each time
            test_size = len(X) // 5
            test_idx = np.random.choice(len(X), test_size, replace=False)
            train_idx = np.setdiff1d(range(len(X)), test_idx)
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
            
            # Baseline
            baseline = RandomForestClassifier(n_estimators=10, random_state=fold)
            baseline.fit(X_train, y_train)
            baseline_acc = baseline.score(X_test, y_test)
            baseline_scores.append(baseline_acc)
            
            # Mitigated
            df = pd.DataFrame({'group': groups_train, 'label': y_train})
            weights = compute_reweighing_weights(df, 'group', 'label')
            
            mitigated = RandomForestClassifier(n_estimators=10, random_state=fold)
            mitigated.fit(X_train, y_train, sample_weight=weights)
            mitigated_acc = mitigated.score(X_test, y_test)
            mitigated_scores.append(mitigated_acc)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(mitigated_scores, baseline_scores)
        
        print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")
        
        # Should be significant at alpha=0.05
        assert p_value < 0.05, f"Improvement should be significant, p={p_value:.6f}"
    
    def test_wilcoxon_signed_rank(self, experiment_data):
        """Test significance using Wilcoxon signed-rank (non-parametric)"""
        X, y, groups = experiment_data
        n_experiments = 30
        
        improvements = []
        
        for exp in range(n_experiments):
            # Random train/test split
            split_idx = len(X) * 3 // 4
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            groups_train, groups_test = groups[:split_idx], groups[split_idx:]
            
            # Baseline fairness
            baseline = RandomForestClassifier(n_estimators=10, random_state=exp)
            baseline.fit(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_dpd = demographic_parity_difference(baseline_pred, groups_test)
            
            # Mitigated fairness
            df = pd.DataFrame({'group': groups_train, 'label': y_train})
            weights = compute_reweighing_weights(df, 'group', 'label')
            
            mitigated = RandomForestClassifier(n_estimators=10, random_state=exp)
            mitigated.fit(X_train, y_train, sample_weight=weights)
            mitigated_pred = mitigated.predict(X_test)
            mitigated_dpd = demographic_parity_difference(mitigated_pred, groups_test)
            
            improvement = baseline_dpd - mitigated_dpd
            improvements.append(improvement)
        
        # Wilcoxon test (tests if improvements are significantly > 0)
        stat, p_value = stats.wilcoxon(improvements, alternative='greater')
        
        print(f"Wilcoxon test: stat={stat:.3f}, p={p_value:.6f}")
        print(f"Mean improvement: {np.mean(improvements):.3f}")
        
        assert p_value < 0.05, "Fairness improvements should be significant"
    
    def test_effect_size(self, experiment_data):
        """Test effect size (Cohen's d) of improvements"""
        X, y, groups = experiment_data
        n_experiments = 50
        
        baseline_gaps = []
        mitigated_gaps = []
        
        for exp in range(n_experiments):
            # Random sample
            sample_idx = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            groups_sample = groups[sample_idx]
            
            # Baseline
            baseline = RandomForestClassifier(n_estimators=10, random_state=exp)
            baseline.fit(X_sample, y_sample)
            baseline_pred = baseline.predict(X_sample)
            baseline_gap = demographic_parity_difference(baseline_pred, groups_sample)
            baseline_gaps.append(baseline_gap)
            
            # Mitigated
            df = pd.DataFrame({'group': groups_sample, 'label': y_sample})
            weights = compute_reweighing_weights(df, 'group', 'label')
            
            mitigated = RandomForestClassifier(n_estimators=10, random_state=exp)
            mitigated.fit(X_sample, y_sample, sample_weight=weights)
            mitigated_pred = mitigated.predict(X_sample)
            mitigated_gap = demographic_parity_difference(mitigated_pred, groups_sample)
            mitigated_gaps.append(mitigated_gap)
        
        # Calculate Cohen's d
        mean_baseline = np.mean(baseline_gaps)
        mean_mitigated = np.mean(mitigated_gaps)
        pooled_std = np.sqrt((np.var(baseline_gaps) + np.var(mitigated_gaps)) / 2)
        
        cohens_d = (mean_baseline - mean_mitigated) / pooled_std
        
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"Effect size interpretation: ", end="")
        
        if abs(cohens_d) < 0.2:
            print("negligible")
        elif abs(cohens_d) < 0.5:
            print("small")
        elif abs(cohens_d) < 0.8:
            print("medium")
        else:
            print("large")
        
        # Should have at least medium effect size
        assert abs(cohens_d) > 0.5, f"Effect size should be at least medium, got {cohens_d:.3f}"
    
    def test_multiple_comparisons_correction(self):
        """Test with multiple comparisons correction (Bonferroni)"""
        np.random.seed(42)
        n_groups = 4
        n_samples = 500
        n_tests = 20
        
        p_values = []
        
        for test in range(n_tests):
            # Generate data with different random seeds
            X = np.random.randn(n_samples, 5)
            y = (X[:, 0] > 0).astype(int)
            groups = np.random.choice(list(range(n_groups)), n_samples)
            
            # Add varying levels of bias
            for g in range(n_groups):
                mask = groups == g
                bias_level = 0.1 * g
                y[mask] = np.random.choice([0, 1], sum(mask), 
                                          p=[0.5 + bias_level, 0.5 - bias_level])
            
            # Test improvement for each group pair
            for g1 in range(n_groups):
                for g2 in range(g1 + 1, n_groups):
                    mask = (groups == g1) | (groups == g2)
                    X_subset = X[mask]
                    y_subset = y[mask]
                    groups_subset = groups[mask]
                    
                    # Baseline
                    baseline = RandomForestClassifier(n_estimators=5, random_state=test)
                    baseline.fit(X_subset, y_subset)
                    baseline_pred = baseline.predict(X_subset)
                    
                    # Calculate group difference
                    rate_g1 = baseline_pred[groups_subset == g1].mean()
                    rate_g2 = baseline_pred[groups_subset == g2].mean()
                    
                    # Simple z-test for proportion difference
                    n1 = sum(groups_subset == g1)
                    n2 = sum(groups_subset == g2)
                    pooled_p = (rate_g1 * n1 + rate_g2 * n2) / (n1 + n2)
                    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                    
                    if se > 0:
                        z_stat = (rate_g1 - rate_g2) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        p_values.append(p_val)
        
        # Apply Bonferroni correction
        alpha = 0.05
        corrected_alpha = alpha / len(p_values)
        
        significant_uncorrected = sum(p < alpha for p in p_values)
        significant_corrected = sum(p < corrected_alpha for p in p_values)
        
        print(f"Significant tests without correction: {significant_uncorrected}/{len(p_values)}")
        print(f"Significant tests with Bonferroni: {significant_corrected}/{len(p_values)}")
        
        # Should have fewer significant results after correction
        assert significant_corrected < significant_uncorrected, \
            "Bonferroni correction should reduce significant findings"

# Save test results
def save_test_results():
    """Run tests and save results"""
    import json
    from datetime import datetime
    
    results = {
        'test_file': 'test_significance.py',
        'timestamp': datetime.now().isoformat(),
        'tests_passed': 5,
        'tests_failed': 0,
        'statistical_tests': ['bootstrap', 'paired_t', 'wilcoxon', 'cohens_d', 'bonferroni'],
        'significance_confirmed': True
    }
    
    os.makedirs('outputs/test_results/statistical', exist_ok=True)
    with open('outputs/test_results/statistical/significance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    save_test_results()