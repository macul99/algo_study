"""
Model Development and Evaluation Test Code
Data Science Framework 2024 - CodeSignal

This module tests comprehensive model evaluation techniques including:
- Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Confusion Matrix analysis
- Regression metrics (MSE, RMSE, MAE, R²)
- Model comparison and selection
- Hyperparameter tuning (Grid Search, Random Search)
- Learning curves and validation curves
- Bias-Variance tradeoff
- Model persistence (saving/loading)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                      GridSearchCV, RandomizedSearchCV,
                                      learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluationTest:
    """
    Comprehensive test class for Model Development and Evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def test_classification_metrics(self):
        """Test 1: Classification Metrics - Comprehensive Evaluation"""
        print("\n" + "="*60)
        print("TEST 1: Classification Metrics")
        print("="*60)
        
        # Generate sample classification data
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nClassification Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\nClass Distribution:")
        print(f"Training:  {np.bincount(y_train)}")
        print(f"Test:      {np.bincount(y_test)}")
        print(f"Predicted: {np.bincount(y_pred)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def test_confusion_matrix(self):
        """Test 2: Confusion Matrix Analysis"""
        print("\n" + "="*60)
        print("TEST 2: Confusion Matrix Analysis")
        print("="*60)
        
        # Generate multi-class classification data
        np.random.seed(42)
        X = np.random.randn(400, 8)
        y = np.random.randint(0, 3, 400)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Calculate per-class metrics
        print(f"\nPer-Class Metrics:")
        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return cm
    
    def test_roc_auc_analysis(self):
        """Test 3: ROC Curve and AUC Score"""
        print("\n" + "="*60)
        print("TEST 3: ROC Curve and AUC Score")
        print("="*60)
        
        # Generate binary classification data
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = ((X[:, 0] + 2*X[:, 1] - X[:, 2]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        print(f"\nROC-AUC Scores:")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC-AUC
            auc_score = roc_auc_score(y_test, y_proba)
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            
            print(f"{name:25s}: {auc_score:.4f}")
            
            # Store results
            self.models[name] = {
                'model': model,
                'auc': auc_score,
                'fpr': fpr,
                'tpr': tpr
            }
        
        return self.models
    
    def test_regression_metrics(self):
        """Test 4: Regression Metrics"""
        print("\n" + "="*60)
        print("TEST 4: Regression Metrics")
        print("="*60)
        
        # Generate regression data
        np.random.seed(42)
        X = np.random.randn(300, 8)
        y = 3*X[:, 0] - 2*X[:, 1] + 1.5*X[:, 2] + np.random.randn(300)*0.5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\nRegression Metrics:")
        print(f"{'Metric':<20s} {'Train':<15s} {'Test':<15s}")
        print(f"{'-'*50}")
        print(f"{'R² Score':<20s} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'MSE':<20s} {train_mse:<15.4f} {test_mse:<15.4f}")
        print(f"{'RMSE':<20s} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE':<20s} {train_mae:<15.4f} {test_mae:<15.4f}")
        
        # Residual analysis
        residuals = y_test - y_pred_test
        print(f"\nResidual Analysis:")
        print(f"Mean Residual: {residuals.mean():.4f}")
        print(f"Std Residual:  {residuals.std():.4f}")
        
        return {
            'r2': test_r2,
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae
        }
    
    def test_cross_validation_detailed(self):
        """Test 5: Detailed Cross-Validation Analysis"""
        print("\n" + "="*60)
        print("TEST 5: Detailed Cross-Validation Analysis")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(300, 10)
        y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
        
        # Test multiple models with cross-validation
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        print(f"\nCross-Validation Results (5-Fold):")
        print(f"{'Model':<25s} {'Mean':<10s} {'Std':<10s} {'Min':<10s} {'Max':<10s}")
        print(f"{'-'*65}")
        
        cv_results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            cv_results[name] = scores
            
            print(f"{name:<25s} {scores.mean():<10.4f} {scores.std():<10.4f} "
                  f"{scores.min():<10.4f} {scores.max():<10.4f}")
        
        return cv_results
    
    def test_grid_search_hyperparameter_tuning(self):
        """Test 6: Grid Search for Hyperparameter Tuning"""
        print("\n" + "="*60)
        print("TEST 6: Grid Search - Hyperparameter Tuning")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(400, 10)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        print(f"\nParameter Grid:")
        for param, values in param_grid.items():
            print(f"{param}: {values}")
        
        # Perform Grid Search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Results
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Test best model
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print(f"Test Score: {test_score:.4f}")
        
        # Top 5 configurations
        print(f"\nTop 5 Configurations:")
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_5 = results_df.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]
        for idx, row in top_5.iterrows():
            print(f"Rank {int(row['rank_test_score'])}: "
                  f"Score={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f}), "
                  f"Params={row['params']}")
        
        return grid_search
    
    def test_random_search_hyperparameter_tuning(self):
        """Test 7: Random Search for Hyperparameter Tuning"""
        print("\n" + "="*60)
        print("TEST 7: Random Search - Hyperparameter Tuning")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(400, 10)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Define parameter distributions
        param_distributions = {
            'n_estimators': [10, 30, 50, 70, 100],
            'max_depth': [3, 5, 7, 9, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8]
        }
        
        print(f"\nParameter Distributions:")
        for param, values in param_distributions.items():
            print(f"{param}: {values}")
        
        # Perform Random Search
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=5, 
            scoring='accuracy', random_state=42, n_jobs=-1, verbose=0
        )
        random_search.fit(X_train, y_train)
        
        # Results
        print(f"\nBest Parameters: {random_search.best_params_}")
        print(f"Best CV Score: {random_search.best_score_:.4f}")
        
        # Test best model
        best_model = random_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print(f"Test Score: {test_score:.4f}")
        
        return random_search
    
    def test_regularization_techniques(self):
        """Test 8: Regularization Techniques (Ridge, Lasso)"""
        print("\n" + "="*60)
        print("TEST 8: Regularization Techniques")
        print("="*60)
        
        # Generate regression data with multicollinearity
        np.random.seed(42)
        X = np.random.randn(300, 20)
        y = 3*X[:, 0] - 2*X[:, 1] + 1.5*X[:, 2] + np.random.randn(300)*0.5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different regularization techniques
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Ridge (α=10.0)': Ridge(alpha=10.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Lasso (α=1.0)': Lasso(alpha=1.0)
        }
        
        print(f"\nRegularization Comparison:")
        print(f"{'Model':<25s} {'Train R²':<12s} {'Test R²':<12s} {'Non-zero Coef':<15s}")
        print(f"{'-'*70}")
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            train_r2 = model.score(X_train_scaled, y_train)
            test_r2 = model.score(X_test_scaled, y_test)
            
            # Count non-zero coefficients
            non_zero = np.sum(np.abs(model.coef_) > 1e-5)
            
            print(f"{name:<25s} {train_r2:<12.4f} {test_r2:<12.4f} {non_zero:<15d}")
        
        return models
    
    def test_bias_variance_tradeoff(self):
        """Test 9: Bias-Variance Tradeoff Analysis"""
        print("\n" + "="*60)
        print("TEST 9: Bias-Variance Tradeoff")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(300, 5)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Test models with different complexity
        max_depths = [1, 3, 5, 10, 20, None]
        
        print(f"\nBias-Variance Tradeoff (Decision Trees):")
        print(f"{'Max Depth':<15s} {'Train Acc':<12s} {'Test Acc':<12s} {'Gap':<12s}")
        print(f"{'-'*51}")
        
        for depth in max_depths:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            gap = train_acc - test_acc
            
            depth_str = str(depth) if depth is not None else 'None'
            print(f"{depth_str:<15s} {train_acc:<12.4f} {test_acc:<12.4f} {gap:<12.4f}")
        
        print(f"\nInterpretation:")
        print(f"- High gap → Overfitting (High Variance)")
        print(f"- Low train & test accuracy → Underfitting (High Bias)")
        print(f"- Optimal: Good test accuracy with small gap")
        
        return None
    
    def test_model_comparison(self):
        """Test 10: Comprehensive Model Comparison"""
        print("\n" + "="*60)
        print("TEST 10: Comprehensive Model Comparison")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(500, 15)
        y = ((X[:, 0] + 2*X[:, 1] - X[:, 2]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        print(f"\nModel Comparison:")
        print(f"{'Model':<25s} {'Accuracy':<12s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s}")
        print(f"{'-'*73}")
        
        results = {}
        for name, model in models.items():
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"{name:<25s} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        print(f"\nBest Model (by F1-Score): {best_model[0]}")
        
        return results
    
    def test_model_persistence(self):
        """Test 11: Model Saving and Loading"""
        print("\n" + "="*60)
        print("TEST 11: Model Persistence (Save/Load)")
        print("="*60)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(200, 8)
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions before saving
        y_pred_before = model.predict(X_test)
        accuracy_before = accuracy_score(y_test, y_pred_before)
        
        print(f"\nOriginal Model Accuracy: {accuracy_before:.4f}")
        
        # Save model
        model_filename = '/tmp/trained_model.joblib'
        joblib.dump(model, model_filename)
        print(f"\nModel saved to: {model_filename}")
        
        # Load model
        loaded_model = joblib.load(model_filename)
        print(f"Model loaded from: {model_filename}")
        
        # Get predictions after loading
        y_pred_after = loaded_model.predict(X_test)
        accuracy_after = accuracy_score(y_test, y_pred_after)
        
        print(f"\nLoaded Model Accuracy: {accuracy_after:.4f}")
        
        # Verify predictions are identical
        predictions_match = np.array_equal(y_pred_before, y_pred_after)
        print(f"Predictions Match: {predictions_match}")
        
        return loaded_model
    
    def run_all_tests(self):
        """Run all model evaluation tests"""
        print("\n" + "="*60)
        print("MODEL DEVELOPMENT AND EVALUATION - COMPREHENSIVE TEST")
        print("="*60)
        
        results = {}
        
        # Run all tests
        results['classification_metrics'] = self.test_classification_metrics()
        results['confusion_matrix'] = self.test_confusion_matrix()
        results['roc_auc'] = self.test_roc_auc_analysis()
        results['regression_metrics'] = self.test_regression_metrics()
        results['cross_validation'] = self.test_cross_validation_detailed()
        results['grid_search'] = self.test_grid_search_hyperparameter_tuning()
        results['random_search'] = self.test_random_search_hyperparameter_tuning()
        results['regularization'] = self.test_regularization_techniques()
        results['bias_variance'] = self.test_bias_variance_tradeoff()
        results['model_comparison'] = self.test_model_comparison()
        results['model_persistence'] = self.test_model_persistence()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results


def main():
    """Main function to run all tests"""
    tester = ModelEvaluationTest()
    results = tester.run_all_tests()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Classification Metrics (Accuracy, Precision, Recall, F1)")
    print("✓ Confusion Matrix Analysis")
    print("✓ ROC Curve and AUC Score")
    print("✓ Regression Metrics (MSE, RMSE, MAE, R²)")
    print("✓ Cross-Validation")
    print("✓ Grid Search Hyperparameter Tuning")
    print("✓ Random Search Hyperparameter Tuning")
    print("✓ Regularization Techniques (Ridge, Lasso)")
    print("✓ Bias-Variance Tradeoff")
    print("✓ Comprehensive Model Comparison")
    print("✓ Model Persistence (Save/Load)")
    print("="*60)


if __name__ == "__main__":
    main()
