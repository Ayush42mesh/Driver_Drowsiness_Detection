import sys
import os

# Add root folder to sys.path to find config.py, model_loader.py, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app import app
