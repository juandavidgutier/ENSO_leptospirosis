import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from zepid.graphics import EffectMeasurePlot

# Set seeds for reproducibility
def seed_everything(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
seed = 999
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set display options for pandas
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create DataFrame for ATE results
df_ATE = pd.DataFrame(0.0, index=range(0, 3), columns=['ATE', '95% CI']).astype({'ATE': 'float64'})
df_ATE['95% CI'] = [((0.0, 0.0)) for _ in range(3)]
print("Initial ATE DataFrame:")
print(df_ATE)

# Import data
data = pd.read_csv("D:/data_leptos.csv", encoding='latin-1')

data = data[['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI',
             'MP', 'sro', 'stl1', 'swvl1', 't2m', 'tp', 'pop_density', 'excess', 
             'NeutralvsLa_Nina', 'NeutralvsEl_Nino', 'El_NinovsLa_Nina']]

# Convert columns to binary
columns_convert = ['SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI']
for col in columns_convert:
    mediana = data[col].median()
    data[col] = (data[col] > mediana).astype(int)

# Function to perform analysis for each comparison
def perform_analysis(data, treatment_col, comparison_name, ate_index):
    print(f"\n{'='*60}")
    print(f"Analysis for {comparison_name}")
    print(f"{'='*60}")
    
    # Prepare data
    data_analysis = data[['excess', treatment_col, 'SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI']]
    data_analysis = data_analysis.dropna()
    
    Y = data_analysis.excess.to_numpy()
    T = data_analysis[treatment_col].to_numpy()
    W = data_analysis[['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI']].to_numpy().reshape(-1, 9)
    X = data_analysis[['SST12']].to_numpy()
    
    # Split the data
    X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
        X, T, Y, W, test_size=0.2, random_state=123)
    
    # Calculate propensity scores and overlap weights for training set only
    logit_model = LogisticRegressionCV(
        penalty='l2', cv=5, random_state=123,
        max_iter=30000, solver='liblinear', scoring='neg_log_loss'
    )
    logit_model.fit(W_train, T_train)
    propensity_scores_train = logit_model.predict_proba(W_train)[:, 1]
    overlap_weights_train = T_train * (1 - propensity_scores_train) + (1 - T_train) * propensity_scores_train
    
    # Estimation of the effect
    estimate = SparseLinearDRLearner(
        featurizer=PolynomialFeatures(degree=3, include_bias=False), 
        max_iter=30000, 
        cv=5, 
        random_state=123)
    
    estimate = estimate.dowhy
    
    # Fit the model with corrected overlap weights
    estimate.fit(
        Y=Y_train, 
        T=T_train, 
        X=X_train, 
        W=W_train, 
        inference='auto', 
        sample_weight=overlap_weights_train
    )
    
    # Predict effect for each sample
    te_pred = estimate.effect(X_test)
    
    # Calculate ATE
    ate = estimate.ate(X_test)
    print(f"ATE for {comparison_name}: {ate}")
    
    # Confidence interval of ATE
    ci = estimate.ate_interval(X_test)
    print(f"95% CI for {comparison_name}: {ci}")
    
    # Set values in df_ATE
    df_ATE.at[ate_index, 'ATE'] = ate
    df_ATE.at[ate_index, '95% CI'] = ci
    
    # CATE Analysis
    max_value = max(X)
    min_value = min(X)
    
    min_X = min_value
    max_X = max_value
    delta = (max_X - min_X) / 100
    X_test_cate = np.arange(min_X, max_X + delta - 0.001, delta).reshape(-1, 1)
    
    est2 = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False), 
                                max_iter=30000, 
                                cv=5, 
                                random_state=123)
    
    est2.fit(Y=Y_train, 
            T=T_train, 
            X=X_train, 
            W=W_train, 
            inference='auto', 
            sample_weight=overlap_weights_train)
    
    treatment_cont_marg = est2.const_marginal_effect(X_test_cate)
    hte_lower_cons, hte_upper_cons = est2.const_marginal_effect_interval(X_test_cate)
    
    # Reshape arrays to 1-dimensional
    X_test_cate = X_test_cate.ravel()
    treatment_cont_marg = treatment_cont_marg.ravel()
    hte_lower_cons = hte_lower_cons.ravel()
    hte_upper_cons = hte_upper_cons.ravel()
    
    # Create CATE plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_cate, treatment_cont_marg, 'b-', linewidth=2, label='Treatment Effect')
    plt.fill_between(X_test_cate, hte_lower_cons, hte_upper_cons, alpha=0.3, color='blue', label='95% CI')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('SST12 °C', fontsize=12)
    plt.ylabel(f'Effect of {comparison_name} on excess leptospirosis cases', fontsize=12)
    plt.title(f'Conditional Average Treatment Effect - {comparison_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Refutation tests
    print(f"\n--- Refutation Tests for {comparison_name} ---")
    
    # Random common cause refutation
    print("\n1. Random Common Cause Refutation:")
    print("-" * 40)
    random_refutation = estimate.refute_estimate(
        method_name="random_common_cause", 
        random_state=123, 
        num_simulations=10
    )
    print(random_refutation)
    
    # Bootstrap refutation
    print("\n2. Bootstrap Refutation:")
    print("-" * 40)
    bootstrap_refutation = estimate.refute_estimate(
        method_name="bootstrap_refuter", 
        random_state=123, 
        num_simulations=10
    )
    print(bootstrap_refutation)
    
    # Dummy outcome refutation
    print("\n3. Dummy Outcome Refutation:")
    print("-" * 40)
    dummy_refutation = estimate.refute_estimate(
        method_name="dummy_outcome_refuter", 
        random_state=123, 
        num_simulations=10
    )
    print(dummy_refutation[0])
    
    # Placebo treatment refutation
    print("\n4. Placebo Treatment Refutation:")
    print("-" * 40)
    placebo_refutation = estimate.refute_estimate(
        method_name="placebo_treatment_refuter", 
        placebo_type="permute", 
        random_state=123, 
        num_simulations=10
    )
    print(placebo_refutation)
    
    return estimate

# Perform analysis for all three comparisons
print("Starting comprehensive analysis for all comparisons...")

# Analysis 1: Neutral vs La Niña
estimate_Neutral_Nina = perform_analysis(data, 'NeutralvsLa_Nina', 'Neutral vs La Niña', 0)

# Analysis 2: Neutral vs El Niño
estimate_Neutral_Nino = perform_analysis(data, 'NeutralvsEl_Nino', 'Neutral vs El Niño', 1)

# Analysis 3: El Niño vs La Niña
estimate_Nino_Nina = perform_analysis(data, 'El_NinovsLa_Nina', 'El Niño vs La Niña', 2)

# Display final ATE results
print(f"\n{'='*60}")
print("FINAL ATE RESULTS")
print(f"{'='*60}")
print(df_ATE)

# Create ATE forest plot
labs = ['Neutral vs La Niña',
        'Neutral vs El Niño',
        'El Niño vs La Niña']

df_labs = pd.DataFrame({'Labels': labs})

print("\nATE Summary:")
print(df_ATE)

# Convert ATE to separate DataFrame
ATE = df_ATE[['ATE']].round(5)
print("\nATE values:")
print(ATE)

# Convert tuples in the '95% CI' column to separate DataFrame
df_ci = df_ATE['95% CI'].apply(pd.Series)

# Rename columns in df_ci
df_ci.columns = ['Lower', 'Upper']

# Create two separate DataFrames for Lower and Upper
Lower = df_ci[['Lower']].copy()
print("\nLower CI bounds:")
print(Lower)
Upper = df_ci[['Upper']].copy()
print("\nUpper CI bounds:")
print(Upper)

# Combine all data for plotting
df_plot = pd.concat([df_labs.reset_index(drop=True), ATE, Lower, Upper], axis=1)
print("\nFinal plotting dataframe:")
print(df_plot)

# Create forest plot for ATE results
print("\nGenerating ATE Forest Plot...")
p = EffectMeasurePlot(label=df_plot.Labels, effect_measure=df_plot.ATE, lcl=df_plot.Lower, ucl=df_plot.Upper)
p.labels(center=0)
p.colors(pointcolor='r', pointshape="s", linecolor='b')
p.labels(effectmeasure='ATE')  
p.plot(figsize=(12, 6), t_adjuster=0.10, max_value=0.12, min_value=-0.05)
plt.tight_layout()
plt.show()

print("\nAnalysis completed successfully!")
print(f"{'='*60}")