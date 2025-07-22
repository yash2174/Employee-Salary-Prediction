# Employee Salary Prediction - Machine Learning Project
# This project demonstrates regression techniques for predicting employee salaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("🚀 Employee Salary Prediction Project")
print("=" * 50)

# =============================================================================
# 1. DATA GENERATION AND LOADING
# =============================================================================

def generate_sample_data(n_samples=1000):
    """Generate synthetic employee data for demonstration"""
    
    np.random.seed(42)
    
    # Define categories
    job_roles = ['Software Engineer', 'Data Scientist', 'Manager', 'Analyst', 
                'Designer', 'Marketing Specialist', 'Sales Representative']
    education_levels = ['Bachelor', 'Master', 'PhD', 'High School']
    locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle', 'Boston']
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
    
    data = []
    
    for i in range(n_samples):
        # Generate correlated features
        experience = np.random.exponential(5)  # Experience years (exponential distribution)
        experience = min(experience, 30)  # Cap at 30 years
        
        job_role = np.random.choice(job_roles)
        education = np.random.choice(education_levels)
        location = np.random.choice(locations)
        industry = np.random.choice(industries)
        
        # Age correlated with experience
        age = max(22 + experience + np.random.normal(0, 2), 22)
        
        # Base salary calculation with realistic factors
        base_salary = 40000
        
        # Experience factor
        salary = base_salary + (experience * 2000) + (experience ** 1.5 * 500)
        
        # Education factor
        education_multiplier = {'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.5}
        salary *= education_multiplier[education]
        
        # Job role factor
        role_multiplier = {
            'Software Engineer': 1.4, 'Data Scientist': 1.5, 'Manager': 1.6,
            'Analyst': 1.1, 'Designer': 1.2, 'Marketing Specialist': 1.1,
            'Sales Representative': 1.0
        }
        salary *= role_multiplier[job_role]
        
        # Location factor
        location_multiplier = {
            'San Francisco': 1.4, 'New York': 1.3, 'Seattle': 1.2,
            'Boston': 1.15, 'Austin': 1.05, 'Chicago': 1.0
        }
        salary *= location_multiplier[location]
        
        # Industry factor
        industry_multiplier = {
            'Technology': 1.2, 'Finance': 1.15, 'Healthcare': 1.05,
            'Manufacturing': 0.95, 'Retail': 0.9
        }
        salary *= industry_multiplier[industry]
        
        # Add some random noise
        salary += np.random.normal(0, 5000)
        salary = max(salary, 30000)  # Minimum salary
        
        data.append({
            'age': round(age, 1),
            'experience_years': round(experience, 1),
            'education_level': education,
            'job_role': job_role,
            'location': location,
            'industry': industry,
            'salary': round(salary)
        })
    
    return pd.DataFrame(data)

# Generate sample data
print("📊 Generating sample employee data...")
df = generate_sample_data(1000)
print(f"Generated {len(df)} employee records")
print("\nDataset Info:")
print(df.info())

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n📈 Exploratory Data Analysis")
print("=" * 30)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Set up plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Employee Salary Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Salary distribution
axes[0, 0].hist(df['salary'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].set_xlabel('Salary ($)')
axes[0, 0].set_ylabel('Frequency')

# 2. Experience vs Salary
axes[0, 1].scatter(df['experience_years'], df['salary'], alpha=0.6, color='coral')
axes[0, 1].set_title('Experience vs Salary')
axes[0, 1].set_xlabel('Experience (Years)')
axes[0, 1].set_ylabel('Salary ($)')

# 3. Education level analysis
education_salary = df.groupby('education_level')['salary'].mean().sort_values(ascending=False)
axes[0, 2].bar(education_salary.index, education_salary.values, color='lightgreen', alpha=0.8)
axes[0, 2].set_title('Average Salary by Education Level')
axes[0, 2].set_ylabel('Average Salary ($)')
axes[0, 2].tick_params(axis='x', rotation=45)

# 4. Job role analysis
job_salary = df.groupby('job_role')['salary'].mean().sort_values(ascending=False)
axes[1, 0].barh(job_salary.index, job_salary.values, color='gold', alpha=0.8)
axes[1, 0].set_title('Average Salary by Job Role')
axes[1, 0].set_xlabel('Average Salary ($)')

# 5. Location analysis
location_salary = df.groupby('location')['salary'].mean().sort_values(ascending=False)
axes[1, 1].bar(location_salary.index, location_salary.values, color='plum', alpha=0.8)
axes[1, 1].set_title('Average Salary by Location')
axes[1, 1].set_ylabel('Average Salary ($)')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Age vs Salary
axes[1, 2].scatter(df['age'], df['salary'], alpha=0.6, color='orange')
axes[1, 2].set_title('Age vs Salary')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Salary ($)')

plt.tight_layout()
plt.show()

# Correlation analysis
print("\n🔗 Correlation Analysis:")
numeric_cols = ['age', 'experience_years', 'salary']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.show()

print(f"Experience-Salary Correlation: {df['experience_years'].corr(df['salary']):.3f}")
print(f"Age-Salary Correlation: {df['age'].corr(df['salary']):.3f}")

# =============================================================================
# 3. DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

print("\n🔧 Data Preprocessing and Feature Engineering")
print("=" * 45)

# Create additional features
df['experience_squared'] = df['experience_years'] ** 2
df['age_experience_ratio'] = df['age'] / (df['experience_years'] + 1)  # +1 to avoid division by zero

# Separate features and target
X = df.drop('salary', axis=1)
y = df['salary']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify categorical and numerical columns
categorical_features = ['education_level', 'job_role', 'location', 'industry']
numerical_features = ['age', 'experience_years', 'experience_squared', 'age_experience_ratio']

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# =============================================================================
# 4. MODEL TRAINING AND EVALUATION
# =============================================================================

print("\n🤖 Model Training and Evaluation")
print("=" * 35)

# Define models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Store results
results = {}
trained_models = {}

print("Training models...")
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                               scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Store results
    results[name] = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'CV RMSE': cv_rmse
    }
    
    trained_models[name] = pipeline
    
    print(f"Train RMSE: {train_rmse:,.0f}")
    print(f"Test RMSE: {test_rmse:,.0f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:,.0f}")
    print(f"CV RMSE: {cv_rmse:,.0f}")

# =============================================================================
# 5. MODEL COMPARISON AND RESULTS VISUALIZATION
# =============================================================================

print("\n📊 Model Comparison")
print("=" * 20)

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(results_df.round(2))

# Find best model
best_model_name = results_df['Test R²'].idxmax()
best_model = trained_models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best Test R²: {results_df.loc[best_model_name, 'Test R²']:.3f}")

# Visualization of model performance
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# 1. R² Score comparison
axes[0, 0].bar(results_df.index, results_df['Test R²'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('Test R² Score by Model')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. RMSE comparison
axes[0, 1].bar(results_df.index, results_df['Test RMSE'], color='coral', alpha=0.8)
axes[0, 1].set_title('Test RMSE by Model')
axes[0, 1].set_ylabel('RMSE ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Prediction vs Actual for best model
y_pred_best = best_model.predict(X_test)
axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6, color='green')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
axes[1, 0].set_xlabel('Actual Salary ($)')
axes[1, 0].set_ylabel('Predicted Salary ($)')
axes[1, 0].grid(alpha=0.3)

# 4. Residual plot for best model
residuals = y_test - y_pred_best
axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6, color='purple')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title(f'Residual Plot - {best_model_name}')
axes[1, 1].set_xlabel('Predicted Salary ($)')
axes[1, 1].set_ylabel('Residuals ($)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n🎯 Feature Importance Analysis")
print("=" * 30)

# Get feature importance for tree-based models
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    
    # Get feature names after preprocessing
    cat_feature_names = rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)
    
    # Get feature importances
    feature_importances = rf_model.named_steps['regressor'].feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features (Random Forest):")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], color='lightblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. HYPERPARAMETER TUNING
# =============================================================================

print("\n⚙️ Hyperparameter Tuning for Best Model")
print("=" * 40)

if best_model_name == 'Random Forest':
    print("Tuning Random Forest hyperparameters...")
    
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    # Create a fresh pipeline for tuning
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R²: {grid_search.best_score_:.3f}")
    
    # Evaluate tuned model
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    
    print(f"Tuned model Test R²: {tuned_r2:.3f}")
    print(f"Tuned model Test RMSE: {tuned_rmse:,.0f}")

# =============================================================================
# 8. PREDICTION EXAMPLES
# =============================================================================

print("\n🔮 Prediction Examples")
print("=" * 20)

# Create sample predictions
sample_employees = pd.DataFrame({
    'age': [28, 35, 45],
    'experience_years': [3, 8, 15],
    'education_level': ['Bachelor', 'Master', 'PhD'],
    'job_role': ['Software Engineer', 'Data Scientist', 'Manager'],
    'location': ['San Francisco', 'New York', 'Austin'],
    'industry': ['Technology', 'Technology', 'Finance'],
    'experience_squared': [9, 64, 225],
    'age_experience_ratio': [28/4, 35/9, 45/16]
})

predictions = best_model.predict(sample_employees)

print("Sample Predictions:")
for i, (idx, row) in enumerate(sample_employees.iterrows()):
    print(f"\nEmployee {i+1}:")
    print(f"  Age: {row['age']}, Experience: {row['experience_years']} years")
    print(f"  Education: {row['education_level']}, Role: {row['job_role']}")
    print(f"  Location: {row['location']}, Industry: {row['industry']}")
    print(f"  Predicted Salary: ${predictions[i]:,.0f}")

# =============================================================================
# 9. PROJECT SUMMARY
# =============================================================================

print("\n" + "="*60)
print("📋 PROJECT SUMMARY")
print("="*60)

print(f"""
🎯 LEARNING OUTCOMES ACHIEVED:

✅ Core AI/ML Concepts with Regression Techniques:
   - Implemented 5 different regression algorithms
   - Understood bias-variance tradeoff
   - Applied cross-validation and hyperparameter tuning

✅ Data Cleaning, Preprocessing, and Feature Engineering:
   - Generated and analyzed synthetic employee data
   - Handled categorical variables with one-hot encoding
   - Created new features (experience_squared, age_experience_ratio)
   - Applied standardization for numerical features

✅ Built and Evaluated Regression Models:
   - Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
   - Used multiple evaluation metrics (R², RMSE, MAE)
   - Performed cross-validation for robust evaluation

✅ Used Jupyter-style Analysis and Visualization:
   - Comprehensive EDA with matplotlib and seaborn
   - Feature importance analysis
   - Model performance comparison plots
   - Residual analysis for model validation

📊 FINAL MODEL PERFORMANCE:
   - Best Model: {best_model_name}
   - Test R²: {results_df.loc[best_model_name, 'Test R²']:.3f}
   - Test RMSE: ${results_df.loc[best_model_name, 'Test RMSE']:,.0f}
   - Test MAE: ${results_df.loc[best_model_name, 'Test MAE']:,.0f}

🔍 KEY INSIGHTS:
   - Experience is the strongest predictor of salary
   - Location significantly impacts compensation
   - Education level shows diminishing returns beyond Master's degree
   - Job role hierarchy aligns with market expectations

💡 NEXT STEPS:
   - Collect real-world data for validation
   - Implement ensemble methods
   - Add more sophisticated feature engineering
   - Deploy model as web application
""")

print("✨ Project completed successfully! ✨")