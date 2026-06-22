import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import clean_and_normalize

def test_clean_and_normalize():
    raw_array = np.array([
        [10.0,  20.0, -9999.0],
        [10.0,  30.0, -9999.0],
        [10.0,  10.0,  10.0]
    ], dtype=np.float32)
    
    nodata_val = -9999.0
    fill_val = 0.0
    
    processed = clean_and_normalize(raw_array, nodata_val, fill_value=fill_val)
    
    assert processed[0, 2] == 0.0
    assert processed[1, 2] == 0.0
    
    assert processed[0, 0] == -1.0
    
    assert processed[1, 1] == 1.0
    
    expected_mid_value = (np.sqrt(10.0) / np.sqrt(20.0)) * 2.0 - 1.0
    
    assert np.isclose(processed[0, 1], expected_mid_value), f"Expected {expected_mid_value}, got {processed[0, 1]}"