"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.utils.helpers import save_model, load_model

logger = setup_logger(__name__, "logs/preprocessor.log")


class DataPreprocessor:
    """Preprocess cancer prediction data"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataPreprocessor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        logger.info("DataPreprocessor initialized")
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values (mean, median, mode, drop)
            
        Returns:
            DataFrame with missing values handled
        """
        strategy = strategy or self.config.get('preprocessing.handle_missing', 'mean')
        df_clean = df.copy()
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            dropped = initial_rows - len(df_clean)
            logger.info(f"Dropped {dropped} rows with missing values")
        else:
            # Handle numerical columns
            numerical_cols = self.config.get('features.numerical', [])
            for col in numerical_cols:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif strategy == 'median':
                        fill_value = df_clean[col].median()
                    else:
                        fill_value = df_clean[col].mean()
                    
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {col} with {strategy}: {fill_value:.2f}")
            
            # Handle categorical columns with mode
            categorical_cols = self.config.get('features.categorical', [])
            for col in categorical_cols:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    fill_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {col} with mode: {fill_value}")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        else:
            logger.info("No duplicates found")
        
        return df_clean
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with dropped columns
        """
        cols_to_drop = self.config.get('features.drop_columns', [])
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if cols_to_drop:
            df_clean = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
        else:
            df_clean = df.copy()
            logger.info("No columns to drop")
        
        return df_clean
    
    def encode_binary_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode binary categorical features (Yes/No) to 1/0
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoder or use existing
            
        Returns:
            DataFrame with encoded binary features
        """
        df_encoded = df.copy()
        
        binary_columns = [
            'Family_History', 'Smoking_History', 'Alcohol_Consumption',
            'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
            'Early_Detection', 'Survival_5_years', 'Mortality',
            'Survival_Prediction'
        ]
        
        for col in binary_columns:
            if col in df_encoded.columns:
                if fit:
                    # Simple mapping for binary Yes/No
                    df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
                    logger.info(f"Encoded binary column: {col}")
                else:
                    df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
        
        return df_encoded
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame,
        fit: bool = True,
        encoding_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoder or use existing
            encoding_type: Type of encoding (onehot, label, target)
            
        Returns:
            DataFrame with encoded features
        """
        encoding_type = encoding_type or self.config.get('preprocessing.encoding', 'onehot')
        df_encoded = df.copy()
        
        # Get categorical columns (excluding binary ones and target)
        categorical_cols = self.config.get('features.categorical', [])
        binary_cols = [
            'Family_History', 'Smoking_History', 'Alcohol_Consumption',
            'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
            'Early_Detection', 'Survival_5_years', 'Mortality',
            'Survival_Prediction'
        ]
        
        # Auto-detect categorical columns (object/category dtype) created by feature engineering
        auto_categorical = [col for col in df_encoded.columns 
                          if df_encoded[col].dtype in ['object', 'category']
                          and col not in binary_cols
                          and col != self.config.get('target_column', 'Survival_Prediction')]
        
        # Combine configured and auto-detected categorical columns
        categorical_cols = list(set(categorical_cols + auto_categorical))
        categorical_cols = [col for col in categorical_cols if col in df_encoded.columns]
        
        if encoding_type == 'onehot':
            logger.info(f"One-hot encoding columns: {categorical_cols}")
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
        
        elif encoding_type == 'label':
            for col in categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col])
                    self.encoders[col] = encoder
                    logger.info(f"Label encoded column: {col}")
                else:
                    if col in self.encoders:
                        df_encoded[col] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        scaling_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler or use existing
            scaling_type: Type of scaling (standard, minmax, robust)
            
        Returns:
            DataFrame with scaled features
        """
        scaling_type = scaling_type or self.config.get('preprocessing.scaling', 'standard')
        df_scaled = df.copy()
        
        numerical_cols = self.config.get('features.numerical', [])
        numerical_cols = [col for col in numerical_cols if col in df_scaled.columns]
        
        if not numerical_cols:
            logger.info("No numerical columns to scale")
            return df_scaled
        
        # Select scaler
        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if fit:
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.scalers['numerical'] = scaler
            logger.info(f"Fitted and scaled numerical features with {scaling_type} scaler")
        else:
            if 'numerical' in self.scalers:
                df_scaled[numerical_cols] = self.scalers['numerical'].transform(df_scaled[numerical_cols])
                logger.info(f"Scaled numerical features with existing scaler")
        
        return df_scaled
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers or use existing
            include_target: Whether to keep target column
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Drop unnecessary columns
        df_clean = self.drop_columns(df)
        
        # 2. Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # 3. Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # 4. Encode binary features
        df_clean = self.encode_binary_features(df_clean, fit=fit)
        
        # 5. Separate target if needed
        target_col = self.config.get('features.target')
        if target_col in df_clean.columns:
            target = df_clean[target_col].copy()
            df_features = df_clean.drop(columns=[target_col])
        else:
            target = None
            df_features = df_clean.copy()
        
        # 6. Encode categorical features
        df_features = self.encode_categorical_features(df_features, fit=fit)
        
        # 7. Scale numerical features
        df_features = self.scale_numerical_features(df_features, fit=fit)
        
        # 8. Add target back if needed
        if target is not None and include_target:
            df_features[target_col] = target.values
        
        # Store feature names
        if fit:
            self.feature_names = [col for col in df_features.columns if col != target_col]
        
        logger.info(f"Preprocessing complete. Final shape: {df_features.shape}")
        
        return df_features
    
    def save_preprocessors(self, path: str = "models/artifacts/preprocessors.pkl") -> None:
        """
        Save fitted preprocessors
        
        Args:
            path: Path to save preprocessors
        """
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }
        save_model(preprocessors, path)
        logger.info(f"Preprocessors saved to: {path}")
    
    def load_preprocessors(self, path: str = "models/artifacts/preprocessors.pkl") -> None:
        """
        Load fitted preprocessors
        
        Args:
            path: Path to load preprocessors from
        """
        preprocessors = load_model(path)
        self.scalers = preprocessors['scalers']
        self.encoders = preprocessors['encoders']
        self.feature_names = preprocessors['feature_names']
        logger.info(f"Preprocessors loaded from: {path}")


if __name__ == "__main__":
    from src.data.data_loader import DataLoader
    
    # Test the preprocessor
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df, fit=True)
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING TEST")
    print("="*80)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"\nProcessed columns: {list(df_processed.columns[:10])}...")
    print(f"\nFirst few rows:")
    print(df_processed.head())
