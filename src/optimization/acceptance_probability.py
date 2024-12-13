import math
from typing import Tuple
from src.fuzzy.fuzzy_number import TriangularFuzzyNumber

def acceptance_probability(current_cost: TriangularFuzzyNumber, 
                         new_cost: TriangularFuzzyNumber, 
                         temperature: float,
                         min_improvement: float = 0.01) -> float:
    """
    Calculate the acceptance probability using modified Metropolis criterion
    with fuzzy cost comparison.
    
    Parameters:
    - current_cost: TriangularFuzzyNumber for current solution cost
    - new_cost: TriangularFuzzyNumber for new solution cost
    - temperature: Current temperature in the SA algorithm
    - min_improvement: Minimum improvement threshold (default 1%)
    
    Returns:
    - float: Acceptance probability between 0 and 1
    """
    # Get fuzzy dominance degree
    dominance = fuzzy_dominance(new_cost, current_cost)
    
    # If new solution clearly dominates current (better)
    if dominance > 0.8:  # Strong dominance
        return 1.0
    
    # If current solution clearly dominates new (worse)
    if dominance < 0.2:  # Weak dominance
        # Calculate acceptance probability based on how much worse it is
        cost_diff = new_cost.defuzzify() - current_cost.defuzzify()
        return math.exp(-cost_diff / (temperature + 1e-10))  # Added small constant to prevent division by zero
    
    # For solutions with similar costs (0.2 <= dominance <= 0.8)
    # Consider accepting based on temperature and relative improvement
    relative_improvement = (current_cost.defuzzify() - new_cost.defuzzify()) / current_cost.defuzzify()
    
    if relative_improvement > min_improvement:
        return 0.5 + (relative_improvement / 2)  # Higher probability for better improvements
    else:
        return 0.5 * math.exp(-abs(relative_improvement) / temperature)

def fuzzy_dominance(cost1: TriangularFuzzyNumber, cost2: TriangularFuzzyNumber) -> float:
    """Calculate the degree to which cost1 dominates cost2."""
    # Handle zero or invalid cases
    if cost1.peak == 0 and cost2.peak == 0:
        return 0.5  # Equal dominance
    if cost1.peak == 0:
        return 0.0  # cost2 dominates
    if cost2.peak == 0:
        return 1.0  # cost1 dominates
    
    # Calculate overlap area
    overlap_start = max(cost1.left, cost2.left)
    overlap_end = min(cost1.right, cost2.right)
    overlap = max(0, overlap_end - overlap_start)
    
    # Calculate total range (handle zero case)
    total_range = max(cost1.right, cost2.right) - min(cost1.left, cost2.left)
    if total_range == 0:
        return 0.5  # Equal dominance
    
    overlap_weight = overlap / total_range
    return 1 - overlap_weight

def possibility_degree(a: TriangularFuzzyNumber, 
                      b: TriangularFuzzyNumber) -> float:
    """
    Calculate the possibility degree that fuzzy number a is less than or equal to b.
    
    Parameters:
    - a: First fuzzy number
    - b: Second fuzzy number
    
    Returns:
    - float: Possibility degree between 0 and 1
    """
    # If a is completely less than b
    if a.right <= b.left:
        return 1.0
    
    # If b is completely less than a
    if b.right <= a.left:
        return 0.0
    
    # Calculate intersection point
    if a.peak == b.peak:
        return 0.5
    
    # Calculate possibility degree based on the relative positions
    # of the triangular fuzzy numbers
    if a.peak <= b.peak:
        return max(0, min(1, (b.right - a.left) / 
                         ((a.right - a.left) + (b.right - b.left))))
    else:
        return max(0, min(1, (b.right - a.left) / 
                         ((a.right - a.left) + (b.right - b.left))))