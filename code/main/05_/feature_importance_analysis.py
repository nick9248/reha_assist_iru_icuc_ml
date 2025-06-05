"""
Feature importance analysis using SHAP values for the NBE prediction model.
Provides detailed insights into feature contributions for model predictions.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_data(datasets_path, logger):
    """
    Load the prepared datasets

    Parameters:
    -----------
    datasets_path : str
        Path to the prepared datasets
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Dictionary containing datasets
    """
    logger.info(f"Loading datasets from {datasets_path}")
    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)
    return datasets


def load_model(model_path, logger):
    """
    Load the refined model

    Parameters:
    -----------
    model_path : str
        Path to the model
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    object
        Trained model
    """
    logger.info(f"Loading model from {model_path}")
    try:
        # Try loading with pickle first
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # If model is a dictionary, extract the model object
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        return model
    except:
        # If that fails, try joblib
        logger.info("Pickle loading failed, trying joblib...")
        model = joblib.load(model_path)
        return model


def compute_shap_values(model, X, feature_names, logger, n_samples=None):
    """
    Compute SHAP values for the model

    Parameters:
    -----------
    model : object
        Trained model
    X : pandas.DataFrame
        Feature matrix
    feature_names : list
        Feature names
    logger : logging.Logger
        Logger instance
    n_samples : int
        Number of samples to use for SHAP calculation (None for all)

    Returns:
    --------
    shap.Explanation or list
        SHAP values explanation object
    """
    logger.info("Computing SHAP values")

    # Extract the model from pipeline if needed
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_component = model.named_steps['model']
        # If the extracted component is a CalibratedClassifierCV, get the base estimator
        if hasattr(model_component, 'base_estimator'):
            model_component = model_component.base_estimator
    else:
        model_component = model

    # Sample data if needed (for large datasets)
    if n_samples is not None and n_samples < X.shape[0]:
        X_sample = X.sample(n_samples, random_state=42)
        logger.info(f"Using {n_samples} random samples for SHAP computation")
    else:
        X_sample = X
        logger.info(f"Using all {X.shape[0]} samples for SHAP computation")

    # Choose the appropriate explainer based on model type
    if isinstance(model_component, RandomForestClassifier):
        logger.info("Using TreeExplainer for RandomForestClassifier")
        explainer = shap.TreeExplainer(model_component)

        try:
            # Try the new style SHAP API which returns an Explanation object
            shap_values = explainer(X_sample)
            logger.info(f"SHAP new API returned type: {type(shap_values)}")
            if hasattr(shap_values, 'values'):
                logger.info(f"SHAP values shape: {shap_values.values.shape}")
        except Exception as e:
            logger.info(f"New SHAP API failed: {str(e)}, trying older API")

            # Fall back to the older SHAP API for trees
            shap_values = explainer.shap_values(X_sample)
            logger.info(f"SHAP values type from old API: {type(shap_values)}")
            if isinstance(shap_values, list):
                logger.info(f"SHAP values is a list of length {len(shap_values)}")
                for i, sv in enumerate(shap_values):
                    logger.info(f"  Class {i} shape: {sv.shape}")
    else:
        logger.info("Using KernelExplainer as fallback")
        # For non-tree models or more complex pipelines, use KernelExplainer
        # Create a function that returns probabilities for class 1
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        # Sample background data
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(predict_fn, background)

        # For KernelExplainer, always use the old API
        shap_values = explainer.shap_values(X_sample)
        logger.info(f"KernelExplainer returned shape: {np.array(shap_values).shape}")

    logger.info("SHAP values computed successfully")
    return shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, feature_names, output_dir, logger):
    """
    Create SHAP summary plot

    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pandas.DataFrame
        Feature matrix used for SHAP computation
    feature_names : list
        Feature names
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating SHAP summary plot")

    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Also create summary plot with only bars (without distribution)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"SHAP summary plots saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
        logger.info("Trying alternative approach")

        try:
            # Alternative approach for different SHAP versions/formats
            plt.figure(figsize=(12, 10))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Use class 1 for binary classification
                shap.summary_plot(shap_values[1], X_sample, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Alternative SHAP summary plot saved to {output_dir}")
        except Exception as e2:
            logger.error(f"Alternative approach also failed: {str(e2)}")
            logger.info("Skipping SHAP summary plot")


def plot_shap_dependence(shap_values, X_sample, feature_names, output_dir, logger, top_n=10):
    """
    Create SHAP dependence plots for top features

    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pandas.DataFrame
        Feature matrix used for SHAP computation
    feature_names : list
        Feature names
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    top_n : int
        Number of top features to plot
    """
    logger.info(f"Creating SHAP dependence plots for top {top_n} features")

    try:
        # First get the feature importances to identify top features
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # For 3D SHAP values (samples, features, classes) - focus on class 1 (NBE=1)
            values = shap_values.values
            feature_importances = np.mean(np.abs(values[:, :, 1]), axis=0)
            n_features = len(feature_importances)
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            # For binary classification with tree models (list format)
            feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
            n_features = len(feature_importances)
        elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2:
            # For 2D SHAP values
            feature_importances = np.mean(np.abs(shap_values.values), axis=0)
            n_features = len(feature_importances)
        else:
            # Other formats - try to extract importances
            if isinstance(shap_values, np.ndarray):
                feature_importances = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(
                    shap_values)
                n_features = len(feature_importances)
            else:
                logger.error(f"Unknown SHAP values format: {type(shap_values)}")
                return

        # Check if feature names match the number of features
        if len(feature_names) != n_features:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match feature importances length ({n_features})")

            if len(X_sample.columns) == n_features:
                logger.info("Using X_sample columns as feature names")
                feature_names = X_sample.columns.tolist()
            else:
                logger.info("Creating generic feature names")
                feature_names = [f"Feature_{i}" for i in range(n_features)]

        # Get indices of top features
        top_indices = np.argsort(feature_importances)[-top_n:]
        logger.info(f"Top {top_n} feature indices: {top_indices}")

        # Plot dependence plots for top features
        for i, idx in enumerate(reversed(top_indices)):
            plt.figure(figsize=(12, 6))
            feature_name = feature_names[idx]
            logger.info(f"  Creating dependence plot for {feature_name} (index {idx})")

            try:
                # For 3D SHAP values - use values for class 1 (NBE=1)
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                    # Extract values for class 1 (NBE=1)
                    class_values = shap_values.values[:, :, 1]

                    # Create scatter plot
                    plt.scatter(X_sample.iloc[:, idx], class_values[:, idx], alpha=0.5)
                    plt.xlabel(feature_name)
                    plt.ylabel(f'SHAP value for {feature_name}')
                    plt.title(f'SHAP Dependence Plot: {feature_name}')

                    # Add trend line
                    z = np.polyfit(X_sample.iloc[:, idx], class_values[:, idx], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(X_sample.iloc[:, idx].min(), X_sample.iloc[:, idx].max(), 100)
                    plt.plot(x_range, p(x_range), "r--", alpha=0.8)

                # For list format SHAP values
                elif isinstance(shap_values, list) and len(shap_values) == 2:
                    # Use class 1 (NBE=1) values
                    class_values = shap_values[1]

                    # Create scatter plot
                    plt.scatter(X_sample.iloc[:, idx], class_values[:, idx], alpha=0.5)
                    plt.xlabel(feature_name)
                    plt.ylabel(f'SHAP value for {feature_name}')
                    plt.title(f'SHAP Dependence Plot: {feature_name}')

                    # Add trend line
                    z = np.polyfit(X_sample.iloc[:, idx], class_values[:, idx], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(X_sample.iloc[:, idx].min(), X_sample.iloc[:, idx].max(), 100)
                    plt.plot(x_range, p(x_range), "r--", alpha=0.8)

                # For 2D SHAP values
                else:
                    # Try to use shap's dependence_plot function
                    try:
                        shap.dependence_plot(
                            idx, shap_values.values if hasattr(shap_values, 'values') else shap_values,
                            X_sample, feature_names=feature_names, show=False
                        )
                    except:
                        # Fall back to simple scatter plot
                        if hasattr(shap_values, 'values'):
                            values = shap_values.values
                        else:
                            values = shap_values

                        plt.scatter(X_sample.iloc[:, idx], values[:, idx], alpha=0.5)
                        plt.xlabel(feature_name)
                        plt.ylabel(f'SHAP value for {feature_name}')
                        plt.title(f'SHAP Dependence Plot: {feature_name}')

                        # Add trend line
                        z = np.polyfit(X_sample.iloc[:, idx], values[:, idx], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(X_sample.iloc[:, idx].min(), X_sample.iloc[:, idx].max(), 100)
                        plt.plot(x_range, p(x_range), "r--", alpha=0.8)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature_name.replace(" ", "_")}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                logger.error(f"Error creating dependence plot for {feature_name}: {str(e)}")
                plt.close()

        logger.info(f"SHAP dependence plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in dependence plot creation: {str(e)}")
        logger.info("Skipping dependence plots")

def plot_top_feature_distributions(X, feature_names, top_shap_features, target, output_dir, logger):
    """
    Plot distributions of top features by target class

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    feature_names : list
        Feature names
    top_shap_features : list
        List of top feature indices from SHAP analysis
    target : pandas.Series
        Target variable
    output_dir : str
        Directory to save plots
    logger : logging.Logger
        Logger instance
    """
    logger.info("Creating distribution plots for top features")

    # Check if we have any top features
    if not top_shap_features:
        logger.warning("No top features provided for distribution plots")
        return

    # Ensure X has the right column names
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)

    # Ensure target and X_df have the same length
    if len(target) != len(X_df):
        logger.warning(f"Target length ({len(target)}) doesn't match X length ({len(X_df)})")
        # Use only as many target values as we have rows in X_df
        target = target.iloc[:len(X_df)] if len(target) > len(X_df) else target
        X_df = X_df.iloc[:len(target)] if len(X_df) > len(target) else X_df

    # Add target column
    X_df['target'] = target.values

    # Get top feature names
    top_feature_names = [feature_names[i] for i in top_shap_features if i < len(feature_names)]

    # Plot distributions
    for feature in top_feature_names:
        if feature not in X_df.columns:
            logger.warning(f"Feature {feature} not found in dataframe")
            continue

        try:
            plt.figure(figsize=(12, 6))

            # Create distribution plot
            sns.histplot(data=X_df, x=feature, hue='target', kde=True,
                         element="step", common_norm=False,
                         palette={0: 'red', 1: 'green'})

            plt.title(f"Distribution of {feature} by Target Class")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend(title='NBE', labels=['Outside NBE (0)', 'Within NBE (1)'])
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'distribution_{feature.replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error creating distribution plot for {feature}: {str(e)}")
            plt.close()

    logger.info(f"Feature distribution plots saved to {output_dir}")


def create_feature_importance_report(shap_values, X_sample, feature_names, output_dir, logger, top_n=20):
    """
    Create feature importance report based on SHAP values

    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pandas.DataFrame
        Feature matrix used for SHAP computation
    feature_names : list
        Feature names
    output_dir : str
        Directory to save report
    logger : logging.Logger
        Logger instance
    top_n : int
        Number of top features to include in report

    Returns:
    --------
    pandas.DataFrame
        Feature importance dataframe
    """
    logger.info("Creating feature importance report")

    # Debug information about shapes
    if isinstance(shap_values, list):
        logger.info(f"SHAP values is a list of length {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            logger.info(f"  Class {i} shape: {sv.shape}")
    elif hasattr(shap_values, 'values'):
        logger.info(f"SHAP Explanation object with values shape: {shap_values.values.shape}")
    else:
        logger.info(f"SHAP values type: {type(shap_values)}")

    # Create feature importance dataframe
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
        # For 3D SHAP values (samples, features, classes)
        logger.info(f"Processing 3D SHAP values with shape {shap_values.values.shape}")

        # Extract values for the positive class (NBE=1, index 1)
        values = shap_values.values
        n_samples, n_features, n_classes = values.shape

        # Average absolute values across samples for class 1 (NBE=1)
        feature_importances = np.mean(np.abs(values[:, :, 1]), axis=0)
        # Average raw values across samples for class 1 (NBE=1)
        feature_impacts = np.mean(values[:, :, 1], axis=0)

        # Check if feature_names matches the number of features
        if len(feature_names) != n_features:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match number of features ({n_features})")
            if len(X_sample.columns) == n_features:
                logger.info("Using X_sample columns as feature names")
                feature_names = X_sample.columns.tolist()
            else:
                logger.info("Creating generic feature names")
                feature_names = [f"Feature_{i}" for i in range(n_features)]

    # For binary classification tree models (older SHAP API)
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        logger.info("Processing binary classification SHAP values (list format)")

        # Number of features should match the second dimension of SHAP values
        n_features = shap_values[0].shape[1]

        # Extract feature importances for class 1 (NBE=1)
        feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
        feature_impacts = np.mean(shap_values[1], axis=0)

        # Ensure feature names match
        if len(feature_names) != n_features:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match number of features ({n_features})")
            if len(X_sample.columns) == n_features:
                logger.info("Using X_sample columns as feature names")
                feature_names = X_sample.columns.tolist()
            else:
                logger.info("Creating generic feature names")
                feature_names = [f"Feature_{i}" for i in range(n_features)]

    # For newer SHAP API (Explanation object with 2D values)
    elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2:
        logger.info("Processing 2D SHAP values")

        values = shap_values.values
        n_features = values.shape[1]

        # Get feature importances
        feature_importances = np.mean(np.abs(values), axis=0)
        feature_impacts = np.mean(values, axis=0)

        # Check feature names
        if len(feature_names) != n_features:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match number of features ({n_features})")
            if len(X_sample.columns) == n_features:
                logger.info("Using X_sample columns as feature names")
                feature_names = X_sample.columns.tolist()
            else:
                logger.info("Creating generic feature names")
                feature_names = [f"Feature_{i}" for i in range(n_features)]

    # For older SHAP versions (single array)
    else:
        logger.info("Processing single SHAP values array")

        if isinstance(shap_values, np.ndarray):
            n_features = shap_values.shape[1] if len(shap_values.shape) > 1 else len(shap_values)
            feature_importances = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(
                shap_values)
            feature_impacts = np.mean(shap_values, axis=0) if len(shap_values.shape) > 1 else shap_values
        else:
            logger.error(f"Unknown SHAP values format: {type(shap_values)}")
            # Return an empty dataframe with feature names
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.nan,
                'Impact': np.nan,
                'Direction': 'Unknown'
            })

        # Check feature names
        if len(feature_names) != n_features:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match number of features ({n_features})")
            if X_sample is not None and len(X_sample.columns) == n_features:
                logger.info("Using X_sample columns as feature names")
                feature_names = X_sample.columns.tolist()
            else:
                logger.info("Creating generic feature names")
                feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Create dataframe with correct feature names and importances
    logger.info(f"Creating importance dataframe with {len(feature_importances)} features")
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],
        'Importance': feature_importances,
        'Impact': feature_impacts
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Calculate percentile importance
    importance_df['Importance_Percentile'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100
    importance_df['Cumulative_Importance'] = importance_df['Importance_Percentile'].cumsum()

    # Determine impact direction (positive/negative)
    importance_df['Direction'] = ['Positive' if x > 0 else 'Negative' for x in importance_df['Impact']]

    # Save to CSV
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # Create top features report
    top_features_df = importance_df.head(top_n)

    # Save top features to CSV
    top_features_df.to_csv(os.path.join(output_dir, 'top_features.csv'), index=False)

    # Calculate feature statistics
    feature_stats = {}
    for feature in top_features_df['Feature'][:min(top_n, len(X_sample.columns))]:
        if feature in X_sample.columns:
            feature_stats[feature] = {
                'mean': X_sample[feature].mean(),
                'std': X_sample[feature].std(),
                'min': X_sample[feature].min(),
                'max': X_sample[feature].max(),
                'median': X_sample[feature].median()
            }

    # Create feature stats dataframe
    if feature_stats:
        stats_df = pd.DataFrame.from_dict(feature_stats, orient='index')
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'Feature'}, inplace=True)

        # Save feature stats to CSV
        stats_df.to_csv(os.path.join(output_dir, 'feature_stats.csv'), index=False)

    logger.info(f"Feature importance report saved to {output_dir}")

    return importance_df

# Add this helper function to extract_direct_feature_importance
def extract_direct_feature_importance(model, feature_names, logger):
    """Extract feature importance directly from model if SHAP fails"""
    logger.info("Extracting feature importance directly from model")

    # Extract the model from pipeline if needed
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_component = model.named_steps['model']
        if hasattr(model_component, 'base_estimator'):
            model_component = model_component.base_estimator
    else:
        model_component = model

    # For tree-based models
    if hasattr(model_component, 'feature_importances_'):
        importances = model_component.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances,
            'Impact': importances,  # Use the same as importance since direction is unknown
            'Direction': ['Unknown'] * len(importances)  # Can't determine direction without SHAP
        }).sort_values('Importance', ascending=False)

        # Add percentile columns
        importance_df['Importance_Percentile'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100
        importance_df['Cumulative_Importance'] = importance_df['Importance_Percentile'].cumsum()

        return importance_df

    # For linear models
    elif hasattr(model_component, 'coef_'):
        coefs = model_component.coef_[0] if len(model_component.coef_.shape) > 1 else model_component.coef_
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(coefs)],
            'Importance': np.abs(coefs),
            'Impact': coefs,
            'Direction': ['Positive' if c > 0 else 'Negative' for c in coefs]
        }).sort_values('Importance', ascending=False)

        # Add percentile columns
        importance_df['Importance_Percentile'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100
        importance_df['Cumulative_Importance'] = importance_df['Importance_Percentile'].cumsum()

        return importance_df

    # Return basic dataframe if can't extract importance
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.nan,
        'Impact': np.nan,
        'Direction': 'Unknown'
    })

def analyze_feature_importance(model, datasets, output_dir, logger, n_samples=None):
    """
    Run complete feature importance analysis

    Parameters:
    -----------
    model : object
        Trained model
    datasets : dict
        Dictionary with datasets
    output_dir : str
        Output directory
    logger : logging.Logger
        Logger instance
    n_samples : int
        Number of samples to use for SHAP calculation

    Returns:
    --------
    pandas.DataFrame
        Feature importance dataframe
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get test data (or validation if not available)
    X_test = datasets.get('X_test_scaled', datasets.get('X_val_scaled',
                                                        datasets.get('X_test', datasets.get('X_val'))))
    y_test = datasets.get('y_test', datasets.get('y_val'))

    # Get feature names
    feature_names = X_test.columns.tolist()

    # Compute SHAP values
    try:
        shap_values, X_sample = compute_shap_values(model, X_test, feature_names, logger, n_samples)
    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")
        logger.info("Trying alternative approach with fewer samples")
        try:
            # Try with fewer samples
            shap_values, X_sample = compute_shap_values(model, X_test, feature_names, logger, 100)
        except Exception as e2:
            logger.error(f"Error computing SHAP values with alternative approach: {str(e2)}")
            logger.info("Feature importance analysis could not be completed")
            return pd.DataFrame({'Feature': feature_names})

    # Create SHAP plots
    try:
        plot_shap_summary(shap_values, X_sample, feature_names, output_dir, logger)
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")

    # Create feature importance report
    try:
        importance_df = create_feature_importance_report(shap_values, X_sample, feature_names, output_dir, logger)
    except Exception as e:
        logger.error(f"Error creating feature importance report: {str(e)}")
        # Create a basic feature importance dataframe
        importance_df = pd.DataFrame({'Feature': feature_names})

    # Create dependence plots
    try:
        plot_shap_dependence(shap_values, X_sample, feature_names, output_dir, logger)
    except Exception as e:
        logger.error(f"Error creating dependence plots: {str(e)}")

    # Create distribution plots for top features
    try:
        if len(importance_df) > 0 and 'Feature' in importance_df.columns:
            top_feature_names = importance_df['Feature'].head(10).tolist()
            top_feature_indices = [feature_names.index(f) for f in top_feature_names if f in feature_names]
            if top_feature_indices:
                plot_top_feature_distributions(X_sample, feature_names, top_feature_indices,
                                               y_test.iloc[:n_samples] if n_samples else y_test, output_dir, logger)
    except Exception as e:
        logger.error(f"Error creating distribution plots: {str(e)}")

    # Save SHAP values for further analysis
    try:
        with open(os.path.join(output_dir, 'shap_values.pkl'), 'wb') as f:
            pickle.dump({
                'shap_values': shap_values,
                'X_sample': X_sample,
                'feature_names': feature_names
            }, f)
    except Exception as e:
        logger.error(f"Error saving SHAP values: {str(e)}")

    logger.info("Feature importance analysis complete")

    return importance_df
if __name__ == "__main__":
    from utils.project_setup import setup_logging, create_project_structure
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--datasets_path", type=str, help="Path to prepared datasets")
    parser.add_argument("--model_path", type=str, help="Path to refined model")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for SHAP calculation")
    args = parser.parse_args()

    # Create project structure
    log_dir, plot_dir, model_dir = create_project_structure()

    # Setup logging
    logger = setup_logging(log_dir, logger_name="feature_importance")
    logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")

    # Find paths if not provided
    if not args.datasets_path or not args.model_path:
        # Try to find the most recent files
        model_base = os.environ.get('MODEL_FOLDER', 'models')
        model_tuning_dir = os.path.join(model_base, "04_model_tuning")

        if os.path.exists(model_tuning_dir):
            subdirs = [os.path.join(model_tuning_dir, d) for d in os.listdir(model_tuning_dir)
                       if os.path.isdir(os.path.join(model_tuning_dir, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)

                # Try to find the refined model
                for model_name in ['refined_model.pkl', 'refined_model_prod.joblib', 'final_model.pkl']:
                    potential_model_path = os.path.join(latest_subdir, '04_model_tuning', model_name)
                    if os.path.exists(potential_model_path):
                        args.model_path = args.model_path or potential_model_path
                        break

                # If we can't find the model in the latest dir, look for prepared datasets
                if not args.datasets_path:
                    # Go back to step 3 for datasets
                    model_dev_dir = os.path.join(model_base, "03_model_development")
                    if os.path.exists(model_dev_dir):
                        dev_subdirs = [os.path.join(model_dev_dir, d) for d in os.listdir(model_dev_dir)
                                       if os.path.isdir(os.path.join(model_dev_dir, d))]
                        if dev_subdirs:
                            latest_dev_subdir = max(dev_subdirs, key=os.path.getmtime)
                            datasets_path = os.path.join(latest_dev_subdir, "prepared_datasets.pkl")
                            if os.path.exists(datasets_path):
                                args.datasets_path = datasets_path

    # Check if paths exist
    if not args.datasets_path or not os.path.exists(args.datasets_path):
        raise ValueError(f"Datasets not found at {args.datasets_path}")
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model not found at {args.model_path}")

    # Log paths
    logger.info(f"Using datasets from: {args.datasets_path}")
    logger.info(f"Using model from: {args.model_path}")

    # Load datasets and model
    datasets = load_data(args.datasets_path, logger)
    model = load_model(args.model_path, logger)

    # Create output directory
    output_dir = os.path.join(model_dir, "feature_importance")
    os.makedirs(output_dir, exist_ok=True)

    # Run feature importance analysis
    importance_df = analyze_feature_importance(model, datasets, output_dir, logger, args.n_samples)

    # Log top 10 important features
    logger.info("Top 10 important features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        logger.info(f"  {i + 1}. {row['Feature']} - Importance: {row['Importance']:.4f}, Direction: {row['Direction']}")

    logger.info("=== FEATURE IMPORTANCE ANALYSIS COMPLETE ===")
    logger.info(f"Results saved to: {output_dir}")