from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    text: str
    tokens: str
    scores: np.array
    explainer: str
    target_pos_idx: int
    helper_type: str
    all_scores: dict
    all_scores2: dict
    target_token_pos_idx: Optional[int] = None
    target: Optional[str] = None
    target_token: Optional[str] = None
    #all_scores: Optional[dict]=None
    

@dataclass
class ExplanationWithRationale(Explanation):
    """Specific explanation to contain the gold rationale"""

    rationale: Optional[np.array] = None
    
    
