from dataclasses import dataclass

@dataclass
class NormalizedHistogramRange:
    """Interval in normalized histogram.

    Attributes
    ----------
    interval_start : float
        Lower bound of the inteval of counted values.
    interval_end : float
        Upper bound of the interval of counted values. Cannot be smaller than interval_start.
    share : float
        The share of values belonging to the intrval in all considered values. Must be between 0 and 1.
    """
    interval_start: float
    interval_end: float
    share: float

    
    
@dataclass
class PredictionStatistics:
    """Statistics of prediction.
    
    Attributes
    ----------
    max_steepnesses : list[float]
        Maximal steepnesses of each street.
    min_turning_radiuses : list[float]
        Minimal turning radiuses of each street.
    """
    max_steepnesses: list[float]
    min_turning_radiuses: list [float]