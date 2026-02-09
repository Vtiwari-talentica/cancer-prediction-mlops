"""Feature engineering utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__, "logs/feature_engineer.log")


class FeatureEngineer:
    """Create new features for cancer prediction"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize FeatureEngineer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        logger.info("FeatureEngineer initialized")
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categories
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age_group feature
        """
        df_new = df.copy()
        
        if 'Age' in df_new.columns:
            df_new['age_group'] = pd.cut(
                df_new['Age'],
                bins=[0, 40, 50, 60, 70, 150],
                labels=['30-40', '40-50', '50-60', '60-70', '70+']
            )
            logger.info("Created age_group feature")
        
        return df_new
    
    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a composite risk score based on multiple risk factors
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk_score feature
        """
        df_new = df.copy()
        
        risk_columns = [
            'Family_History', 'Smoking_History', 'Alcohol_Consumption',
            'Diabetes', 'Genetic_Mutation'
        ]
        
        # Count how many risk factors are present
        available_cols = [col for col in risk_columns if col in df_new.columns]
        
        if available_cols:
            # Convert to binary if not already
            for col in available_cols:
                if df_new[col].dtype == 'object':
                    df_new[col + '_binary'] = (df_new[col] == 'Yes').astype(int)
                    available_cols_binary = [col + '_binary' for col in available_cols]
                else:
                    available_cols_binary = available_cols
            
            df_new['risk_score'] = df_new[available_cols_binary].sum(axis=1)
            logger.info(f"Created risk_score from {len(available_cols)} risk factors")
            
            # Clean up temporary binary columns
            temp_cols = [col + '_binary' for col in available_cols if col + '_binary' in df_new.columns]
            df_new.drop(columns=temp_cols, inplace=True, errors='ignore')
        
        return df_new
    
    def create_lifestyle_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lifestyle health score
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lifestyle_score feature
        """
        df_new = df.copy()
        
        lifestyle_score = 0
        factors_count = 0
        
        # Obesity BMI (worse if obese)
        if 'Obesity_BMI' in df_new.columns:
            bmi_map = {'Normal': 2, 'Overweight': 1, 'Obese': 0}
            df_new['bmi_score'] = df_new['Obesity_BMI'].map(bmi_map)
            lifestyle_score += df_new['bmi_score']
            factors_count += 1
        
        # Diet Risk (worse if high risk)
        if 'Diet_Risk' in df_new.columns:
            diet_map = {'Low': 2, 'Moderate': 1, 'High': 0}
            df_new['diet_score'] = df_new['Diet_Risk'].map(diet_map)
            lifestyle_score += df_new['diet_score']
            factors_count += 1
        
        # Physical Activity (better if high)
        if 'Physical_Activity' in df_new.columns:
            activity_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            df_new['activity_score'] = df_new['Physical_Activity'].map(activity_map)
            lifestyle_score += df_new['activity_score']
            factors_count += 1
        
        if factors_count > 0:
            df_new['lifestyle_score'] = lifestyle_score / factors_count
            logger.info(f"Created lifestyle_score from {factors_count} factors")
            
            # Clean up temporary scores
            df_new.drop(columns=['bmi_score', 'diet_score', 'activity_score'], 
                       inplace=True, errors='ignore')
        
        return df_new
    
    def create_healthcare_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create healthcare quality composite score
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with healthcare_quality_score feature
        """
        df_new = df.copy()
        
        score = 0
        factors_count = 0
        
        # Healthcare Access
        if 'Healthcare_Access' in df_new.columns:
            access_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            df_new['access_score'] = df_new['Healthcare_Access'].map(access_map)
            score += df_new['access_score']
            factors_count += 1
        
        # Insurance Status
        if 'Insurance_Status' in df_new.columns:
            insurance_map = {'Uninsured': 0, 'Insured': 1}
            df_new['insurance_score'] = df_new['Insurance_Status'].map(insurance_map)
            score += df_new['insurance_score']
            factors_count += 1
        
        # Economic Classification
        if 'Economic_Classification' in df_new.columns:
            economic_map = {'Developing': 0, 'Developed': 1}
            df_new['economic_score'] = df_new['Economic_Classification'].map(economic_map)
            score += df_new['economic_score']
            factors_count += 1
        
        # Screening History
        if 'Screening_History' in df_new.columns:
            screening_map = {'Never': 0, 'Irregular': 1, 'Regular': 2}
            df_new['screening_score'] = df_new['Screening_History'].map(screening_map)
            score += df_new['screening_score']
            factors_count += 1
        
        if factors_count > 0:
            df_new['healthcare_quality_score'] = score / factors_count
            logger.info(f"Created healthcare_quality_score from {factors_count} factors")
            
            # Clean up temporary scores
            df_new.drop(columns=['access_score', 'insurance_score', 'economic_score', 'screening_score'],
                       inplace=True, errors='ignore')
        
        return df_new
    
    def create_tumor_severity_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize tumor severity based on size
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with tumor_severity feature
        """
        df_new = df.copy()
        
        if 'Tumor_Size_mm' in df_new.columns:
            # Based on exploration: mean=42mm, median=42mm, range=5-79mm
            df_new['tumor_severity'] = pd.cut(
                df_new['Tumor_Size_mm'],
                bins=[0, 20, 40, 60, 100],
                labels=['Small', 'Medium', 'Large', 'Very_Large']
            )
            logger.info("Created tumor_severity feature")
        
        return df_new
    
    def create_age_tumor_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction between age and tumor size
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age_tumor_interaction feature
        """
        df_new = df.copy()
        
        if 'Age' in df_new.columns and 'Tumor_Size_mm' in df_new.columns:
            # Normalize to avoid large values
            df_new['age_tumor_interaction'] = (
                (df_new['Age'] / df_new['Age'].max()) * 
                (df_new['Tumor_Size_mm'] / df_new['Tumor_Size_mm'].max())
            )
            logger.info("Created age_tumor_interaction feature")
        
        return df_new
    
    def create_cost_per_mm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cost per mm of tumor size
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cost_per_mm feature
        """
        df_new = df.copy()
        
        if 'Healthcare_Costs' in df_new.columns and 'Tumor_Size_mm' in df_new.columns:
            # Avoid division by zero
            df_new['cost_per_mm'] = df_new['Healthcare_Costs'] / (df_new['Tumor_Size_mm'] + 1)
            logger.info("Created cost_per_mm feature")
        
        return df_new
    
    def create_stage_severity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numerical score for cancer stage
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with stage_severity_score feature
        """
        df_new = df.copy()
        
        if 'Cancer_Stage' in df_new.columns:
            stage_map = {'Localized': 1, 'Regional': 2, 'Metastatic': 3}
            df_new['stage_severity_score'] = df_new['Cancer_Stage'].map(stage_map)
            logger.info("Created stage_severity_score feature")
        
        return df_new
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        # Create all features
        df_engineered = self.create_age_groups(df_engineered)
        df_engineered = self.create_risk_score(df_engineered)
        df_engineered = self.create_lifestyle_score(df_engineered)
        df_engineered = self.create_healthcare_quality_score(df_engineered)
        df_engineered = self.create_tumor_severity_category(df_engineered)
        df_engineered = self.create_age_tumor_interaction(df_engineered)
        df_engineered = self.create_cost_per_mm(df_engineered)
        df_engineered = self.create_stage_severity_score(df_engineered)
        
        initial_cols = df.shape[1]
        final_cols = df_engineered.shape[1]
        new_features = final_cols - initial_cols
        
        logger.info(f"Feature engineering complete. Added {new_features} new features")
        logger.info(f"Final shape: {df_engineered.shape}")
        
        return df_engineered


if __name__ == "__main__":
    from src.data.data_loader import DataLoader
    
    # Test feature engineering
    loader = DataLoader()
    df = loader.load_data()
    
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING TEST")
    print("="*80)
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    print(f"\nNew columns:")
    new_cols = set(df_engineered.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
