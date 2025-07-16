from datetime import datetime
import kaggle 
import pandas as pd
import logging
import os
import sys


def extract_key():
    with open('kaggle-token.json', 'r') as f:
        return f.read()

def extract_data_from_kaggle_dataset():
    key = extract_key()
    return ke