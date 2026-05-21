import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import clean_and_normalize

def test_clean_and_normalize():
    """
    Ensures NoData values are ignored during min/max scaling, 
    and then filled with the specified fill_value.
    """
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
    
    assert processed[0, 0] == 0.0
    
    assert processed[1, 1] == 1.0
    
    assert processed[0, 1] == 0.5