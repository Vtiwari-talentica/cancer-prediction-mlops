"""Data loading utilities"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__, "logs/data_loader.log")


class DataLoader:
    """Load and manage cancer prediction dataset"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize DataLoader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.data_path = self.config.get('paths.raw_data')
        logger.info(f"DataLoader initialized with config: {config_path}")
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load cancer prediction data from CSV
        
        Args:
            file_path: Optional custom path to data file
            
        Returns:
            DataFrame with loaded data
        """
        path = file_path if file_path else self.data_path
        
        try:
            logger.info(f"Loading data from: {path}")
            df = pd.read_csv(path)
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        logger.info(f"Dataset info: {info['shape'][0]} rows, {info['shape'][1]} columns, "
                   f"{info['memory_usage']:.2f} MB")
        
        return info
    
    def split_data(
        self, 
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.config.get('data.test_size', 0.2)
        validation_size = validation_size or self.config.get('data.validation_size', 0.1)
        random_state = random_state or self.config.get('data.random_state', 42)
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[self.config.get('features.target')]
        )
        
        # Second split: train and validation
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val[self.config.get('features.target')]
            )
            
            logger.info(f"Data split - Train: {train.shape[0]}, "
                       f"Val: {val.shape[0]}, Test: {test.shape[0]}")
            
            return train, test, val
        else:
            logger.info(f"Data split - Train: {train_val.shape[0]}, "
                       f"Test: {test.shape[0]}")
            return train_val, test, None
    
    def save_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            file_path: Path to save file
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_data()
    info = loader.get_data_info(df)
    
    print("\n" + "="*80)
    print("DATA LOADER TEST")
    print("="*80)
    print(f"\nDataset Shape: {info['shape']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    print(f"\nFirst few rows:")
    print(df.head())
