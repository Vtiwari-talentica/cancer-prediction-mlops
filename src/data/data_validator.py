"""Data validation utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__, "logs/data_validator.log")


class DataValidator:
    """Validate cancer prediction dataset"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataValidator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.expected_columns = (
            self.config.get('features.numerical', []) +
            self.config.get('features.categorical', []) +
            [self.config.get('features.target')] +
            self.config.get('features.drop_columns', [])
        )
        logger.info("DataValidator initialized")
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for unexpected columns
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            issues.append(f"Unexpected columns: {extra_cols}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.warning(f"Schema validation failed: {issues}")
        
        return is_valid, issues
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for missing values in DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of column names and missing counts
        """
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        if missing_dict:
            logger.warning(f"Missing values found: {missing_dict}")
        else:
            logger.info("No missing values found")
        
        return missing_dict
    
    def check_duplicates(self, df: pd.DataFrame) -> int:
        """
        Check for duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return duplicates
    
    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Validate data types of columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of unexpected data types
        """
        issues = {}
        
        # Check numerical columns
        numerical_cols = self.config.get('features.numerical', [])
        for col in numerical_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues[col] = f"Expected numeric, got {df[col].dtype}"
        
        # Check categorical columns
        categorical_cols = self.config.get('features.categorical', [])
        for col in categorical_cols:
            if col in df.columns and not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                # Categorical can be string or object type
                pass
        
        if issues:
            logger.warning(f"Data type issues: {issues}")
        else:
            logger.info("Data types validated successfully")
        
        return issues
    
    def check_value_ranges(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Check if numerical values are within expected ranges
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of range violations
        """
        issues = {}
        
        # Age should be between 30-89 (based on exploration)
        if 'Age' in df.columns:
            if df['Age'].min() < 0 or df['Age'].max() > 150:
                issues['Age'] = f"Age out of range: {df['Age'].min()} - {df['Age'].max()}"
        
        # Tumor size should be positive
        if 'Tumor_Size_mm' in df.columns:
            if df['Tumor_Size_mm'].min() < 0:
                issues['Tumor_Size_mm'] = "Negative tumor sizes found"
        
        # Healthcare costs should be positive
        if 'Healthcare_Costs' in df.columns:
            if df['Healthcare_Costs'].min() < 0:
                issues['Healthcare_Costs'] = "Negative costs found"
        
        if issues:
            logger.warning(f"Value range issues: {issues}")
        else:
            logger.info("Value ranges validated successfully")
        
        return issues
    
    def validate_categorical_values(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Check for unexpected categorical values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of columns with unexpected values
        """
        issues = {}
        
        # Define expected values for key categorical columns
        expected_values = {
            'Gender': ['M', 'F'],
            'Cancer_Stage': ['Localized', 'Regional', 'Metastatic'],
            'Obesity_BMI': ['Normal', 'Overweight', 'Obese'],
            'Diet_Risk': ['Low', 'Moderate', 'High'],
            'Physical_Activity': ['Low', 'Moderate', 'High'],
            'Screening_History': ['Regular', 'Irregular', 'Never'],
            'Urban_or_Rural': ['Urban', 'Rural'],
            'Economic_Classification': ['Developed', 'Developing'],
            'Healthcare_Access': ['Low', 'Moderate', 'High'],
            'Insurance_Status': ['Insured', 'Uninsured'],
            'Survival_Prediction': ['Yes', 'No']
        }
        
        for col, expected in expected_values.items():
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                unexpected = set(unique_values) - set(expected)
                if unexpected:
                    issues[col] = list(unexpected)
        
        if issues:
            logger.warning(f"Unexpected categorical values: {issues}")
        else:
            logger.info("Categorical values validated successfully")
        
        return issues
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive validation report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Generating validation report...")
        
        schema_valid, schema_issues = self.validate_schema(df)
        missing_values = self.check_missing_values(df)
        duplicates = self.check_duplicates(df)
        dtype_issues = self.validate_data_types(df)
        range_issues = self.check_value_ranges(df)
        categorical_issues = self.validate_categorical_values(df)
        
        report = {
            'schema_valid': schema_valid,
            'schema_issues': schema_issues,
            'missing_values': missing_values,
            'duplicate_rows': duplicates,
            'data_type_issues': dtype_issues,
            'value_range_issues': range_issues,
            'categorical_value_issues': categorical_issues,
            'overall_valid': (
                schema_valid and 
                len(missing_values) == 0 and 
                duplicates == 0 and
                len(dtype_issues) == 0 and
                len(range_issues) == 0 and
                len(categorical_issues) == 0
            )
        }
        
        if report['overall_valid']:
            logger.info("✅ All validation checks passed!")
        else:
            logger.warning("⚠️ Some validation checks failed")
        
        return report


if __name__ == "__main__":
    from src.data.data_loader import DataLoader
    
    # Test the validator
    loader = DataLoader()
    df = loader.load_data()
    
    validator = DataValidator()
    report = validator.generate_validation_report(df)
    
    print("\n" + "="*80)
    print("DATA VALIDATION REPORT")
    print("="*80)
    print(f"\nOverall Valid: {report['overall_valid']}")
    print(f"Schema Valid: {report['schema_valid']}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print(f"Missing Values: {len(report['missing_values'])} columns")
    print(f"Data Type Issues: {len(report['data_type_issues'])} columns")
    print(f"Value Range Issues: {len(report['value_range_issues'])} columns")
    print(f"Categorical Issues: {len(report['categorical_value_issues'])} columns")
