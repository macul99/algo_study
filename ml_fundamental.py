"""
Machine Learning Fundamentals Test Code
Data Science Framework 2024 - CodeSignal

This module tests core ML concepts including:
- Data preprocessing and feature engineering
- Train-test splitting and cross-validation
- Basic ML algorithms (Linear Regression, Logistic Regression, Decision Trees, etc.)
- Feature scaling and normalization
- Handling missing data
- Encoding categorical variables
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MLFundamentalsTest:
    """
    Comprehensive test class for Machine Learning Fundamentals
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def test_data_preprocessing(self):
        """Test 1: Data Preprocessing and Cleaning"""
        print("\n" + "="*60)
        print("TEST 1: Data Preprocessing and Cleaning")
        print("="*60)
        
        # Create sample dataset with missing values
        data = {
            'age': [25, 30, np.nan, 35, 40, np.nan, 28],
            'salary': [50000, 60000, 55000, np.nan, 75000, 65000, 58000],
            'department': ['IT', 'HR', 'IT', 'Finance', np.nan, 'IT', 'HR'],
            'experience': [2, 5, 3, 8, 10, 4, 3]
        }
        df = pd.DataFrame(data)
        
        print("\nOriginal Data with Missing Values:")
        print(df)
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Handle missing numerical values - mean imputation
        numerical_imputer = SimpleImputer(strategy='mean')
        df[['age', 'salary']] = numerical_imputer.fit_transform(df[['age', 'salary']])
        
        # Handle missing categorical values - most frequent
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[['department']] = categorical_imputer.fit_transform(df[['department']])
        
        print("\nData After Imputation:")
        print(df)
        print(f"\nMissing values after imputation:\n{df.isnull().sum()}")
        
        return df
    
    def test_feature_scaling(self):
        """Test 2: Feature Scaling and Normalization"""
        print("\n" + "="*60)
        print("TEST 2: Feature Scaling and Normalization")
        print("="*60)
        
        # Sample data
        data = np.array([[1, 100, 0.1],
                        [2, 200, 0.2],
                        [3, 300, 0.3],
                        [4, 400, 0.4],
                        [5, 500, 0.5]])
        
        print("\nOriginal Data:")
        print(data)
        
        # StandardScaler (z-score normalization)
        scaler_standard = StandardScaler()
        data_standard = scaler_standard.fit_transform(data)
        
        print("\nStandardScaler (mean=0, std=1):")
        print(data_standard)
        print(f"Mean: {data_standard.mean(axis=0)}")
        print(f"Std: {data_standard.std(axis=0)}")
        
        # MinMaxScaler (range [0, 1])
        scaler_minmax = MinMaxScaler()
        data_minmax = scaler_minmax.fit_transform(data)
        
        print("\nMinMaxScaler (range [0, 1]):")
        print(data_minmax)
        print(f"Min: {data_minmax.min(axis=0)}")
        print(f"Max: {data_minmax.max(axis=0)}")
        
        return data_standard, data_minmax
    
    def test_encoding_categorical_variables(self):
        """Test 3: Encoding Categorical Variables"""
        print("\n" + "="*60)
        print("TEST 3: Encoding Categorical Variables")
        print("="*60)
        
        # Sample categorical data
        categories = ['Low', 'Medium', 'High', 'Low', 'High', 'Medium']
        colors = ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green']
        
        print(f"\nOriginal categorical data (ordinal): {categories}")
        print(f"Original categorical data (nominal): {colors}")
        
        # Label Encoding (for ordinal data)
        label_encoder = LabelEncoder()
        categories_encoded = label_encoder.fit_transform(categories)
        
        print(f"\nLabel Encoded (ordinal): {categories_encoded}")
        print(f"Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # One-Hot Encoding (for nominal data)
        colors_array = np.array(colors).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        colors_onehot = onehot_encoder.fit_transform(colors_array)
        
        print(f"\nOne-Hot Encoded (nominal):")
        print(colors_onehot)
        print(f"Categories: {onehot_encoder.categories_}")
        
        return categories_encoded, colors_onehot
    
    def test_train_test_split(self):
        """Test 4: Train-Test Split"""
        print("\n" + "="*60)
        print("TEST 4: Train-Test Split")
        print("="*60)
        
        # Generate sample dataset
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data - 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def test_cross_validation(self):
        """Test 5: K-Fold Cross-Validation"""
        print("\n" + "="*60)
        print("TEST 5: K-Fold Cross-Validation")
        print("="*60)
        
        # Generate sample dataset
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Create a simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n5-Fold Cross-Validation Results:")
        print(f"Fold scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")
        print(f"95% Confidence interval: {scores.mean():.4f} (+/- {1.96 * scores.std():.4f})")
        
        return scores
    
    def test_linear_regression(self):
        """Test 6: Linear Regression"""
        print("\n" + "="*60)
        print("TEST 6: Linear Regression")
        print("="*60)
        
        # Generate sample regression data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate R² score
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\nModel Coefficients: {model.coef_}")
        print(f"Model Intercept: {model.intercept_:.4f}")
        print(f"Train R² Score: {train_score:.4f}")
        print(f"Test R² Score: {test_score:.4f}")
        
        return model, test_score
    
    def test_logistic_regression(self):
        """Test 7: Logistic Regression (Binary Classification)"""
        print("\n" + "="*60)
        print("TEST 7: Logistic Regression")
        print("="*60)
        
        # Generate sample classification data
        np.random.seed(42)
        X = np.random.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Coefficients: {model.coef_}")
        print(f"Model Intercept: {model.intercept_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nSample predictions (first 5):")
        print(f"Actual: {y_test[:5]}")
        print(f"Predicted: {y_pred[:5]}")
        print(f"Probabilities:\n{y_proba[:5]}")
        
        return model, accuracy
    
    def test_decision_tree(self):
        """Test 8: Decision Tree Classifier"""
        print("\n" + "="*60)
        print("TEST 8: Decision Tree Classifier")
        print("="*60)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(150, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"\nTree Depth: {model.get_depth()}")
        print(f"Number of Leaves: {model.get_n_leaves()}")
        print(f"Feature Importances: {model.feature_importances_}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return model, test_accuracy
    
    def test_random_forest(self):
        """Test 9: Random Forest Classifier"""
        print("\n" + "="*60)
        print("TEST 9: Random Forest Classifier")
        print("="*60)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = ((X[:, 0] + X[:, 1] > 0) & (X[:, 2] < 0)).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"\nNumber of Trees: {model.n_estimators}")
        print(f"Feature Importances: {model.feature_importances_}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return model, test_accuracy
    
    def test_knn_classifier(self):
        """Test 10: K-Nearest Neighbors Classifier"""
        print("\n" + "="*60)
        print("TEST 10: K-Nearest Neighbors Classifier")
        print("="*60)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(150, 3)
        y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features (important for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print(f"\nNumber of Neighbors: {model.n_neighbors}")
        print(f"Distance Metric: {model.metric}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return model, test_accuracy
    
    def run_all_tests(self):
        """Run all ML fundamentals tests"""
        print("\n" + "="*60)
        print("MACHINE LEARNING FUNDAMENTALS - COMPREHENSIVE TEST")
        print("="*60)
        
        results = {}
        
        # Run all tests
        results['preprocessing'] = self.test_data_preprocessing()
        results['scaling'] = self.test_feature_scaling()
        results['encoding'] = self.test_encoding_categorical_variables()
        results['train_test_split'] = self.test_train_test_split()
        results['cross_validation'] = self.test_cross_validation()
        results['linear_regression'] = self.test_linear_regression()
        results['logistic_regression'] = self.test_logistic_regression()
        results['decision_tree'] = self.test_decision_tree()
        results['random_forest'] = self.test_random_forest()
        results['knn'] = self.test_knn_classifier()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results


def main():
    """Main function to run all tests"""
    tester = MLFundamentalsTest()
    results = tester.run_all_tests()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Data Preprocessing")
    print("✓ Feature Scaling")
    print("✓ Categorical Encoding")
    print("✓ Train-Test Split")
    print("✓ Cross-Validation")
    print("✓ Linear Regression")
    print("✓ Logistic Regression")
    print("✓ Decision Trees")
    print("✓ Random Forests")
    print("✓ K-Nearest Neighbors")
    print("="*60)


if __name__ == "__main__":
    main()
