"""
Data Explorer Module for NBE Prediction Project
Generates comprehensive visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataExplorer:
    """
    Handles exploratory data analysis and visualization generation
    """

    def __init__(self, plots_path: Path, log_path: Path):
        self.plots_path = plots_path
        self.log_path = log_path
        self.logger = self._setup_logger()

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create plots directory
        self.plots_dir = self.plots_path / 'step1_data_exploration'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data exploration operations"""
        logger = logging.getLogger('DataExplorer')
        logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = self.log_path / 'step1'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'data_explorer_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)

        return logger

    def create_data_distribution_plots(self, df: pd.DataFrame) -> str:
        """
        Create comprehensive data distribution visualizations

        Args:
            df: DataFrame to visualize

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating data distribution plots")

        # Define the features to plot
        features = ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe']
        available_features = [f for f in features if f in df.columns]

        if not available_features:
            self.logger.warning("No expected features found in dataframe")
            return None

        # Create subplots
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, feature in enumerate(available_features):
            ax = axes[i]

            # Create count plot
            value_counts = df[feature].value_counts().sort_index()
            bars = ax.bar(value_counts.index, value_counts.values, alpha=0.7)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            ax.set_title(f'{feature.upper()} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

            # Add percentage labels
            total = len(df)
            percentages = [f"{v / total * 100:.1f}%" for v in value_counts.values]
            for j, (bar, pct) in enumerate(zip(bars, percentages)):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() / 2,
                        pct, ha='center', va='center', fontweight='bold', color='white')

        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'data_distribution_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Data distribution plot saved: {plot_path}")
        return str(plot_path)

    def create_correlation_matrix(self, df: pd.DataFrame) -> str:
        """
        Create correlation matrix heatmap for numeric features

        Args:
            df: DataFrame to analyze

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating correlation matrix")

        # Select numeric columns
        numeric_cols = ['p_score', 'p_status', 'fl_score', 'fl_status', 'nbe']
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) < 2:
            self.logger.warning("Not enough numeric columns for correlation analysis")
            return None

        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'correlation_matrix_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Correlation matrix saved: {plot_path}")
        return str(plot_path)

    def create_missing_values_heatmap(self, df: pd.DataFrame) -> str:
        """
        Create heatmap showing missing values pattern

        Args:
            df: DataFrame to analyze

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating missing values heatmap")

        # Calculate missing values
        missing_data = df.isnull()

        if not missing_data.any().any():
            self.logger.info("No missing values found - creating summary plot instead")

            # Create a summary plot showing completeness
            fig, ax = plt.subplots(figsize=(12, 6))

            completeness = (1 - df.isnull().mean()) * 100
            bars = ax.bar(range(len(completeness)), completeness.values, alpha=0.7, color='green')

            ax.set_title('Data Completeness by Column', fontsize=16, fontweight='bold')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Completeness (%)')
            ax.set_xticks(range(len(completeness)))
            ax.set_xticklabels(completeness.index, rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

            # Add percentage labels
            for bar, pct in zip(bars, completeness.values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        else:
            # Create missing values heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Pattern', fontsize=16, fontweight='bold')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows')

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'missing_values_heatmap_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Missing values heatmap saved: {plot_path}")
        return str(plot_path)

    def create_target_variable_analysis(self, df: pd.DataFrame) -> str:
        """
        Create comprehensive analysis of the target variable (nbe)

        Args:
            df: DataFrame containing target variable

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating target variable analysis")

        if 'nbe' not in df.columns:
            self.logger.warning("Target variable 'nbe' not found in dataframe")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Variable (NBE) Analysis', fontsize=16, fontweight='bold')

        # 1. Distribution plot
        ax1 = axes[0, 0]
        value_counts = df['nbe'].value_counts().sort_index()
        bars = ax1.bar(value_counts.index, value_counts.values, alpha=0.7)

        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        ax1.set_title('NBE Distribution', fontweight='bold')
        ax1.set_xlabel('NBE Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)

        # 2. Percentage pie chart
        ax2 = axes[0, 1]
        labels = [f'NBE {i}' for i in value_counts.index]
        percentages = value_counts.values / len(df) * 100

        wedges, texts, autotexts = ax2.pie(value_counts.values, labels=labels, autopct='%1.1f%%',
                                           startangle=90, explode=[0.05] * len(value_counts))
        ax2.set_title('NBE Distribution (%)', fontweight='bold')

        # 3. NBE vs Features heatmap
        ax3 = axes[1, 0]
        features = ['p_score', 'p_status', 'fl_score', 'fl_status']
        available_features = [f for f in features if f in df.columns]

        if available_features:
            # Create cross-tabulation
            feature_nbe_corr = df[available_features + ['nbe']].corr()['nbe'][:-1]

            bars = ax3.barh(range(len(feature_nbe_corr)), feature_nbe_corr.values, alpha=0.7)
            ax3.set_yticks(range(len(feature_nbe_corr)))
            ax3.set_yticklabels(feature_nbe_corr.index)
            ax3.set_xlabel('Correlation with NBE')
            ax3.set_title('Feature-NBE Correlations', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add correlation values
            for i, (bar, corr) in enumerate(zip(bars, feature_nbe_corr.values)):
                ax3.text(bar.get_width() + 0.01 if bar.get_width() >= 0 else bar.get_width() - 0.01,
                         bar.get_y() + bar.get_height() / 2, f'{corr:.3f}',
                         ha='left' if bar.get_width() >= 0 else 'right', va='center', fontweight='bold')

        # 4. Class balance assessment
        ax4 = axes[1, 1]

        # Calculate class imbalance metrics
        class_counts = df['nbe'].value_counts().sort_index()
        class_proportions = class_counts / len(df)

        # Create stacked bar showing balance
        ax4.bar(['Dataset'], [1.0], color='lightgray', alpha=0.3, label='Total')

        bottom = 0
        colors = ['red', 'yellow', 'green']
        for i, (nbe_val, proportion) in enumerate(class_proportions.items()):
            ax4.bar(['Dataset'], [proportion], bottom=bottom,
                    color=colors[i % len(colors)], alpha=0.7,
                    label=f'NBE {nbe_val}: {proportion:.2%}')
            bottom += proportion

        ax4.set_title('Class Balance Assessment', fontweight='bold')
        ax4.set_ylabel('Proportion')
        ax4.legend()
        ax4.set_ylim(0, 1)

        # Add balance status
        min_class_prop = class_proportions.min()
        balance_status = "Balanced" if min_class_prop >= 0.3 else "Imbalanced"
        ax4.text(0, 0.5, f'Status: {balance_status}\nMin class: {min_class_prop:.2%}',
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'target_variable_distribution_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Target variable analysis saved: {plot_path}")
        return str(plot_path)

    def create_patient_consultation_analysis(self, df: pd.DataFrame) -> str:
        """
        Analyze consultation patterns per patient

        Args:
            df: DataFrame with accident_number (patient ID)

        Returns:
            str: Path to saved plot
        """
        self.logger.info("Creating patient consultation analysis")

        if 'accident_number' not in df.columns:
            self.logger.warning("Patient identifier 'accident_number' not found")
            return None

        # Calculate consultations per patient
        consultations_per_patient = df.groupby('accident_number').size()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Patient Consultation Analysis', fontsize=16, fontweight='bold')

        # 1. Distribution of consultations per patient
        ax1 = axes[0, 0]
        consultation_counts = consultations_per_patient.value_counts().sort_index()

        bars = ax1.bar(consultation_counts.index, consultation_counts.values, alpha=0.7)
        ax1.set_title('Consultations per Patient Distribution', fontweight='bold')
        ax1.set_xlabel('Number of Consultations')
        ax1.set_ylabel('Number of Patients')
        ax1.grid(True, alpha=0.3)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # 2. Summary statistics
        ax2 = axes[0, 1]
        stats = consultations_per_patient.describe()

        stats_text = f"""
        Total Patients: {len(consultations_per_patient):,}
        Total Consultations: {consultations_per_patient.sum():,}

        Mean: {stats['mean']:.2f}
        Median: {stats['50%']:.2f}
        Min: {int(stats['min'])}
        Max: {int(stats['max'])}
        Std: {stats['std']:.2f}
        """

        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.set_title('Consultation Statistics', fontweight='bold')
        ax2.axis('off')

        # 3. Boxplot
        ax3 = axes[1, 0]
        box_plot = ax3.boxplot(consultations_per_patient.values, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][0].set_alpha(0.7)

        ax3.set_title('Consultations per Patient (Boxplot)', fontweight='bold')
        ax3.set_ylabel('Number of Consultations')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        sorted_consultations = consultations_per_patient.sort_values()
        cumulative_pct = np.arange(1, len(sorted_consultations) + 1) / len(sorted_consultations) * 100

        ax4.plot(sorted_consultations.values, cumulative_pct, marker='o', markersize=2, alpha=0.7)
        ax4.set_title('Cumulative Distribution', fontweight='bold')
        ax4.set_xlabel('Number of Consultations')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f'patient_consultation_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Patient consultation analysis saved: {plot_path}")
        return str(plot_path)

    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate all exploration plots and return paths

        Args:
            df: DataFrame to analyze

        Returns:
            Dict containing paths to all generated plots
        """
        self.logger.info("Starting comprehensive data exploration")

        plot_paths = {}

        # Generate all plots
        try:
            plot_paths['distribution'] = self.create_data_distribution_plots(df)
            plot_paths['correlation'] = self.create_correlation_matrix(df)
            plot_paths['missing_values'] = self.create_missing_values_heatmap(df)
            plot_paths['target_analysis'] = self.create_target_variable_analysis(df)
            plot_paths['patient_analysis'] = self.create_patient_consultation_analysis(df)

            self.logger.info("All exploration plots generated successfully")

        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise

        return plot_paths