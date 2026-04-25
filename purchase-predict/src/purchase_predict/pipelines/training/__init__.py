"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.2.0
"""

from dotenv import load_dotenv
from .pipeline import create_pipeline

# Cette ligne va lire le fichier .env et charger l'IP de ton serveur MLflow !
load_dotenv()

__all__ = ["create_pipeline"]

__version__ = "0.1"