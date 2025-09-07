import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from dowhy import CausalModel  # <-- Added for causal DAG modeling

# Set seeds for reproducibility
def seed_everything(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
seed = 999
seed_everything(seed)
warnings.filterwarnings('ignore')

#%%
# Set display options for pandas
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create DataFrame for ATE results
df_ATE = pd.DataFrame(0.0, index=range(0, 3), columns=['ATE', '95% CI']).astype({'ATE': 'float64'})
df_ATE['95% CI'] = [((0.0, 0.0)) for _ in range(3)]
print(df_ATE)

#%%

# Import data
data = pd.read_csv("D:/clases/UDES/fortalecimiento institucional/macroproyecto_2023/articulo2/leptos/ci/data_leptos.csv", encoding='latin-1')

# Check and clean duplicate columns
if data.columns.duplicated().any():
    print("Removing duplicate columns...")
    data = data.loc[:, ~data.columns.duplicated()]

data = data[['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI',
             'MPI', 'sro', 'stl1', 'swvl1', 't2m', 'tp', 'pop_density', 'excess', 'cases',
             'NeutralvsLa_Nina', 'NeutralvsEl_Nino', 'El_NinovsLa_Nina', 
             'DANE', 'DANE_year', 'DANE_period']]

# Convert columns to binary
columns_convert = ['SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI']
for col in columns_convert:
    if col in data.columns:
        mediana = data[col].median()
        data[col] = (data[col] > mediana).astype(int)

#%%

# Function to create normalized encodings (without dropna)
def create_normalized_encodings(df):
    df = df.copy()
    
    # DANE (municipality) - encode only columns that exist
    if 'DANE' in df.columns:
        le = LabelEncoder()
        df['DANE_labeled'] = le.fit_transform(df['DANE'])
        scaler = MinMaxScaler()
        df['DANE_normalized'] = scaler.fit_transform(df[['DANE_labeled']])
    
    # DANE_year (annual patterns)
    if 'DANE_year' in df.columns:
        le_year = LabelEncoder()
        df['DANE_year_labeled'] = le_year.fit_transform(df['DANE_year'])
        scaler_year = MinMaxScaler()
        df['DANE_year_normalized'] = scaler_year.fit_transform(df[['DANE_year_labeled']])
    
    # DANE_period (monthly patterns)
    if 'DANE_period' in df.columns:
        le_period = LabelEncoder()
        df['DANE_period_labeled'] = le_period.fit_transform(df['DANE_period'])
        scaler_period = MinMaxScaler()
        df['DANE_period_normalized'] = scaler_period.fit_transform(df[['DANE_period_labeled']])
    
    return df

# Create encoded data without removing NaN
data_encoded = create_normalized_encodings(data)

print(f"Shape of data_encoded: {data_encoded.shape}")
print(f"Columns in data_encoded: {data_encoded.columns.tolist()}")

#%%

# Function to dynamically build causal graph (DOT format) for a given treatment
def build_causal_graph(treatment_col):
    """
    Builds a DOT-format causal graph string, dynamically inserting the treatment column name.
    Assumes variable names in data are mapped to DAG node names (e.g., SST12 -> S12).
    """
    graph_template = f"""
    digraph G {{
        // Nodes: Climate indices, treatment, outcome, mediators, confounders
        S12; S3; S34; S4; SOI; ESOI; NATL; TROP; SATL;
        {treatment_col}; excess;
        sro; stl1; swvl1; t2m; tp;
        MPI; pop_density;

        // Relationships among climate indices
        S12 -> S3; S12 -> S34; S12 -> S4; S12 -> SOI; S12 -> ESOI;
        S12 -> NATL; S12 -> SATL; S12 -> TROP;

        S3 -> S34; S3 -> S4; S3 -> SOI; S3 -> ESOI; S3 -> NATL;
        S3 -> SATL; S3 -> TROP;

        S34 -> S4; S34 -> SOI; S34 -> ESOI; S34 -> NATL; S34 -> SATL; S34 -> TROP;

        S4 -> SOI; S4 -> ESOI; S4 -> NATL; S4 -> SATL; S4 -> TROP;

        SOI -> ESOI; SOI -> NATL; SOI -> SATL; SOI -> TROP;

        ESOI -> NATL; ESOI -> SATL; ESOI -> TROP;

        NATL -> SATL; NATL -> TROP;

        SATL -> TROP;

        // Climate indices -> Treatment
        S12 -> {treatment_col}; S3 -> {treatment_col}; S34 -> {treatment_col};
        S4 -> {treatment_col}; SOI -> {treatment_col}; ESOI -> {treatment_col};
        NATL -> {treatment_col}; SATL -> {treatment_col}; TROP -> {treatment_col};

        // Climate indices -> Outcome
        S12 -> excess; S3 -> excess; S34 -> excess; S4 -> excess;
        SOI -> excess; ESOI -> excess; NATL -> excess; SATL -> excess; TROP -> excess;

        // Climate indices -> Mediators (soil/rain/temp variables)
        S12 -> sro; S12 -> stl1; S12 -> swvl1; S12 -> t2m; S12 -> tp;
        S3 -> sro; S3 -> stl1; S3 -> swvl1; S3 -> t2m; S3 -> tp;
        S34 -> sro; S34 -> stl1; S34 -> swvl1; S34 -> t2m; S34 -> tp;
        S4 -> sro; S4 -> stl1; S4 -> swvl1; S4 -> t2m; S4 -> tp;
        SOI -> sro; SOI -> stl1; SOI -> swvl1; SOI -> t2m; SOI -> tp;
        ESOI -> sro; ESOI -> stl1; ESOI -> swvl1; ESOI -> t2m; ESOI -> tp;
        NATL -> sro; NATL -> stl1; NATL -> swvl1; NATL -> t2m; NATL -> tp;
        SATL -> sro; SATL -> stl1; SATL -> swvl1; SATL -> t2m; SATL -> tp;
        TROP -> sro; TROP -> stl1; TROP -> swvl1; TROP -> t2m; TROP -> tp;

        // Treatment -> Mediators
        {treatment_col} -> sro; {treatment_col} -> stl1; {treatment_col} -> swvl1;
        {treatment_col} -> t2m; {treatment_col} -> tp;

        // Mediators interdependencies
        sro -> stl1; sro -> swvl1; sro -> t2m; sro -> tp;
        stl1 -> swvl1; stl1 -> t2m; stl1 -> tp;
        swvl1 -> t2m; swvl1 -> tp;
        t2m -> tp;

        // Mediators -> MPI, pop_density, outcome
        sro -> MPI; sro -> excess; sro -> pop_density;
        stl1 -> MPI; stl1 -> excess; stl1 -> pop_density;
        swvl1 -> MPI; swvl1 -> excess; swvl1 -> pop_density;
        t2m -> MPI; t2m -> excess; t2m -> pop_density;
        tp -> MPI; tp -> excess; tp -> pop_density;

        // Final confounders / modifiers
        MPI -> excess; pop_density -> excess;
        MPI -> pop_density; 
    }}
    """
    return graph_template


# Function to estimate ATE and CATE for a specific treatment using DoWhy + EconML
def estimate_treatment_effect(data, treatment_col, treatment_name, df_ATE, row_index):
    print(f"\n=== Estimating effect for {treatment_name} ===")
    
    # Create specific dataset for this treatment (apply dropna here)
    required_cols = ['excess', treatment_col, 'SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI', 
                     'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized']
    treatment_data = data[required_cols].copy()
    
    # Remove only rows with NaN in required columns
    treatment_data = treatment_data.dropna()
    print(f"Shape after dropna for {treatment_name}: {treatment_data.shape}")
    
    if treatment_data.shape[0] == 0:
        print(f"No data for {treatment_name} after removing NaN!")
        return df_ATE, None, None
    
    # Prepare arrays for EconML (still needed for CATE and internal fitting)
    Y = treatment_data['excess'].to_numpy()
    T = treatment_data[treatment_col].to_numpy()
    W = treatment_data[['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI']].to_numpy()
    X = treatment_data[['SST12', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized']].to_numpy()  

    # Check dimensions
    print(f"Dimensions - Y: {Y.shape}, T: {T.shape}, W: {W.shape}, X: {X.shape}")

    # Split data
    X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
        X, T, Y, W, test_size=0.2, random_state=999, stratify=T
    )

    # Calculate propensity scores and overlap weights
    logit_model = LogisticRegressionCV(
        penalty='l2', cv=5, random_state=999,
        max_iter=30000, solver='liblinear', scoring='neg_log_loss'
    )
    logit_model.fit(W_train, T_train)
    propensity_scores_train = logit_model.predict_proba(W_train)[:, 1]
    overlap_weights_train = T_train * (1 - propensity_scores_train) + (1 - T_train) * propensity_scores_train
    overlap_weights_train = np.clip(overlap_weights_train, 0.01, 100)
   
    propensity_scores_test = logit_model.predict_proba(W_test)[:, 1]
    overlap_weights_test = T_test * (1 - propensity_scores_test) + (1 - T_test) * propensity_scores_test
    overlap_weights_test = np.clip(overlap_weights_test, 0.01, 100)

    propensity_scores_all = logit_model.predict_proba(W)[:, 1]
    overlap_weights_all = T * (1 - propensity_scores_all) + (1 - T) * propensity_scores_all
    overlap_weights_all = np.clip(overlap_weights_all, 0.01, 100)

    # ================================
    # CAUSAL MODELING WITH DoWhy
    # ================================
    # Prepare data for DoWhy (must include all DAG variables)
    data_for_dowhy = treatment_data.copy()

    
    # Rename columns to match DAG node names
    rename_map = {
        'SST12': 'S12',
        'SST3': 'S3',
        'SST34': 'S34',
        'SST4': 'S4',
        'NATL': 'NATL',
        'SATL': 'SATL',
        'TROP': 'TROP',
        'SOI': 'SOI',
        'ESOI': 'ESOI',
        'sro': 'sro',
        'stl1': 'stl1',
        'swvl1': 'swvl1',
        't2m': 't2m',
        'tp': 'tp',
        'pop_density': 'pop_density',
        'excess': 'excess'
    }
    data_for_dowhy = data_for_dowhy.rename(columns=rename_map)

    # Build causal graph dynamically for this treatment
    graph_str = build_causal_graph(treatment_col)

    # Initialize CausalModel
    model = CausalModel(
        data=data_for_dowhy,
        treatment=[treatment_col],
        outcome=['excess'],
        graph=graph_str
    )

    # Identify causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand (adjustment set):")
    print(identified_estimand)

    # Estimate effect using EconML within DoWhy framework
    estimate_effect = model.estimate_effect(
        identified_estimand,
        effect_modifiers=['S12', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized'],
        method_name="backdoor.econml.dr.SparseLinearDRLearner",
        confidence_intervals=True,
        method_params={
            "init_params": {
                "featurizer": PolynomialFeatures(degree=3, include_bias=False),
                "max_iter": 30000,
                "cv": 5,
                "random_state": 999
            },
            "fit_params": {
                "sample_weight": overlap_weights_all  # Use full sample weights
            }
        }
    )

    # Extract ATE and confidence interval
    ate = estimate_effect.value
    ci_lower, ci_upper = estimate_effect.get_confidence_intervals()[0]
    ci = (ci_lower, ci_upper)
    
    # Store in df_ATE
    df_ATE.at[row_index, 'ATE'] = ate
    df_ATE.at[row_index, '95% CI'] = ci
    
    print(f"ATE {treatment_name}: {ate:.5f}")
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]")

    # ================================
    # CATE: Conditional Effect on SST12 (S12)
    # ================================
    # Extract SST12 values (first column in X)
    SST12_train = X_train[:, 0]  
    SST12_test = X_test[:, 0]    

    # Create grid for SST12
    min_SST12 = SST12_train.min()
    max_SST12 = SST12_train.max()
    delta = (max_SST12 - min_SST12) / 100
    SST12_grid = np.arange(min_SST12, max_SST12 + delta - 0.001, delta)

    # Compute mean values for other effect modifiers
    DANE_encoded_mean = np.mean(X_train[:, 1])   
    DANE_year_encoded_mean = np.mean(X_train[:, 2])
    DANE_period_encoded_mean = np.mean(X_train[:, 3])    

    # Construct X_test_grid: [SST12, DANE_norm, DANE_year_norm, DANE_period_norm]
    X_test_grid = np.column_stack([
        SST12_grid,  
        np.full_like(SST12_grid, DANE_encoded_mean),     
        np.full_like(SST12_grid, DANE_year_encoded_mean),
        np.full_like(SST12_grid, DANE_period_encoded_mean)  
    ])

    # Access internal EconML estimator to compute CATE
    econml_estimator = estimate_effect.estimator.estimator  # This is the actual SparseLinearDRLearner

    treatment_cont_marg = econml_estimator.effect(X_test_grid)
    hte_lower2_cons, hte_upper2_cons = econml_estimator.effect_interval(X_test_grid)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'X_test': SST12_grid,
        'treatment_cont_marg': treatment_cont_marg,
        'hte_lower2_cons': hte_lower2_cons,
        'hte_upper2_cons': hte_upper2_cons
    })

    # Plot CATE
    cate_plot = (
        ggplot(plot_data)
        + aes(x='X_test', y='treatment_cont_marg')
        + geom_line(color='blue', size=1)
        + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
        + labs(x='SST12 °C', y=f'Effect of {treatment_name} on excess leptospirosis cases',
               title=f'CATE - {treatment_name}')
        + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
        + theme(plot_title=element_text(hjust=0.5, size=12),
                axis_title_x=element_text(size=10),
                axis_title_y=element_text(size=10))
    )
    
    # ================================
    # REFUTATION TESTS (using DoWhy's API)
    # ================================
    print(f"\n=== Refutation tests for {treatment_name} ===")
    try:
        # Random Common Cause
        random_test = model.refute_estimate(
            estimated_effect=estimate_effect,
            method_name="random_common_cause",
            random_state=123,
            num_simulations=50
        )
        print(f"Random common cause refutation: {random_test}")

        # Bootstrap
        bootstrap_test = model.refute_estimate(
            estimated_effect=estimate_effect,
            method_name="bootstrap_refuter",
            random_state=123,
            num_simulations=50
        )
        print(f"Bootstrap refutation: {bootstrap_test}")

        # Dummy Outcome
        dummy_test = model.refute_estimate(
            estimated_effect=estimate_effect,
            method_name="dummy_outcome_refuter",
            random_state=123,
            num_simulations=50
        )
        print(f"Dummy outcome refutation: {dummy_test}")

        # Placebo Treatment
        placebo_test = model.refute_estimate(
            estimated_effect=estimate_effect,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            random_state=123,
            num_simulations=50
        )
        print(f"Placebo treatment refutation: {placebo_test}")

    except Exception as e:
        print(f"Error during refutation tests: {e}")

    return df_ATE, cate_plot, estimate_effect

#%%

# Estimate effects for all three treatments
treatments_info = [
    ('NeutralvsLa_Nina', 'Neutral vs La Niña', 0),
    ('NeutralvsEl_Nino', 'Neutral vs El Niño', 1),
    ('El_NinovsLa_Nina', 'El Niño vs La Niña', 2)
]

cate_plots = []
models = []

for treatment_col, treatment_name, row_index in treatments_info:
    if treatment_col in data_encoded.columns:
        df_ATE, cate_plot, model = estimate_treatment_effect(data_encoded, treatment_col, treatment_name, df_ATE, row_index)
        cate_plots.append(cate_plot)
        models.append(model)
        print(f"✓ Completed: {treatment_name}")
    else:
        print(f"✗ Column not found: {treatment_col}")

print("\n=== FINAL RESULTS ===")
print(df_ATE)

#%%

# Figure ATE
labs = ['Neutral vs La Niña',
        'Neutral vs El Niño', 
        'El Niño vs La Niña']

df_labs = pd.DataFrame({'Labels': labs})

print("Final ATE DataFrame:")
print(df_ATE)

# Convert ATE to separate DataFrame
ATE = df_ATE[['ATE']].round(5)
print("ATE values:")
print(ATE)

# Convert tuples in the '95% CI' column to separate DataFrame
df_ci = df_ATE['95% CI'].apply(pd.Series)

# Rename columns in df_ci
df_ci.columns = ['Lower', 'Upper']

# Create two separate DataFrames for Lower and Upper
Lower = df_ci[['Lower']].copy()
print("Lower CI:")
print(Lower)
Upper = df_ci[['Upper']].copy()
print("Upper CI:")
print(Upper)

df_plot = pd.concat([df_labs.reset_index(drop=True), ATE, Lower, Upper], axis=1)
print("DataFrame for plotting:")
print(df_plot)

# Create ATE plot
p = EffectMeasurePlot(label=df_plot.Labels, effect_measure=df_plot.ATE, lcl=df_plot.Lower, ucl=df_plot.Upper)
p.labels(center=0)
p.colors(pointcolor='r', pointshape="s", linecolor='b')
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.10, max_value=0.12, min_value=-0.05)
plt.tight_layout()
plt.title('Average Treatment Effects')
plt.show()

# Show CATE plots for all three treatments
print("\n=== CATE PLOTS ===")

# CATE 1: Neutral vs La Niña
if cate_plots[0] is not None:
    print("1. CATE - Neutral vs La Niña:")
    cate_plots[0].draw()
    plt.show()
else:
    print("1. CATE - Neutral vs La Niña: Not available")

# CATE 2: Neutral vs El Niño  
if cate_plots[1] is not None:
    print("2. CATE - Neutral vs El Niño:")
    cate_plots[1].draw()
    plt.show()
else:
    print("2. CATE - Neutral vs El Niño: Not available")

# CATE 3: El Niño vs La Niña
if cate_plots[2] is not None:
    print("3. CATE - El Niño vs La Niña:")
    cate_plots[2].draw()
    plt.show()
else:
    print("3. CATE - El Niño vs La Niña: Not available")