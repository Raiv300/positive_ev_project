"""
devig_utils.py

Utility functions for odds conversion and de-vigging (vig removal) in sports betting markets.
Used in PGA Outrights betting project for converting bookmaker odds to fair probabilities.

Author: [Your Name]
Date: October 2024
"""

import numpy as np
import pandas as pd
from typing import Union, List


# ============================================================================
# ODDS CONVERSION FUNCTIONS
# ============================================================================

def american_to_decimal(american_odds: Union[int, float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert American odds to decimal (European) odds.
    
    American odds format:
        +150 means bet $100 to win $150 (decimal: 2.50)
        -150 means bet $150 to win $100 (decimal: 1.67)
    
    Decimal odds format:
        Total return per $1 wagered (including stake)
    
    Parameters
    ----------
    american_odds : int, float, np.ndarray, or pd.Series
        American odds (can be positive or negative)
    
    Returns
    -------
    decimal_odds : float, np.ndarray, or pd.Series
        Decimal odds (always >= 1.0)
    
    Examples
    --------
    >>> american_to_decimal(100)
    2.0
    >>> american_to_decimal(-110)
    1.909090909090909
    >>> american_to_decimal([100, -110, 250])
    array([2.        , 1.90909091, 3.5       ])
    """
    # Convert to numpy array for vectorized operations
    american = np.asarray(american_odds, dtype=float)
    
    # Positive odds: decimal = 1 + (american / 100)
    # Negative odds: decimal = 1 + (100 / |american|)
    decimal = np.where(
        american > 0,
        1.0 + (american / 100.0),      # Positive odds
        1.0 + (100.0 / np.abs(american))  # Negative odds
    )
    
    # Return same type as input
    if isinstance(american_odds, pd.Series):
        return pd.Series(decimal, index=american_odds.index)
    elif isinstance(american_odds, (int, float)):
        return float(decimal)
    else:
        return decimal


def decimal_to_american(decimal_odds: Union[float, np.ndarray, pd.Series]) -> Union[int, np.ndarray, pd.Series]:
    """
    Convert decimal (European) odds to American odds.
    
    Parameters
    ----------
    decimal_odds : float, np.ndarray, or pd.Series
        Decimal odds (must be >= 1.0)
    
    Returns
    -------
    american_odds : int, np.ndarray, or pd.Series
        American odds (positive or negative)
    
    Examples
    --------
    >>> decimal_to_american(2.0)
    100
    >>> decimal_to_american(1.91)
    -110
    >>> decimal_to_american([2.0, 1.91, 3.5])
    array([ 100, -110,  250])
    """
    decimal = np.asarray(decimal_odds, dtype=float)
    
    # Validate input
    if np.any(decimal < 1.0):
        raise ValueError("Decimal odds must be >= 1.0")
    
    # If decimal >= 2.0: american = (decimal - 1) * 100
    # If decimal < 2.0: american = -100 / (decimal - 1)
    american = np.where(
        decimal >= 2.0,
        (decimal - 1.0) * 100.0,           # Positive American odds
        -100.0 / (decimal - 1.0)           # Negative American odds
    )
    
    # Round to nearest integer (standard convention)
    american = np.round(american).astype(int)
    
    # Return same type as input
    if isinstance(decimal_odds, pd.Series):
        return pd.Series(american, index=decimal_odds.index)
    elif isinstance(decimal_odds, (int, float)):
        return int(american)
    else:
        return american


def implied_from_american(american_odds: Union[int, float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate implied probability from American odds.
    
    This is the "raw" implied probability BEFORE de-vigging.
    The sum of implied probabilities across all outcomes in a market will be > 1.0
    due to the bookmaker's overround (vig/margin).
    
    Formula: implied_prob = 1 / decimal_odds
    
    Parameters
    ----------
    american_odds : int, float, np.ndarray, or pd.Series
        American odds
    
    Returns
    -------
    implied_prob : float, np.ndarray, or pd.Series
        Implied probability (between 0 and 1)
    
    Examples
    --------
    >>> implied_from_american(100)
    0.5
    >>> implied_from_american(-110)
    0.5238095238095238
    """
    decimal = american_to_decimal(american_odds)
    implied = 1.0 / decimal
    
    # Return same type as input
    if isinstance(american_odds, pd.Series):
        return pd.Series(implied, index=american_odds.index)
    elif isinstance(american_odds, (int, float)):
        return float(implied)
    else:
        return implied


def implied_from_decimal(decimal_odds: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate implied probability from decimal odds.
    
    Parameters
    ----------
    decimal_odds : float, np.ndarray, or pd.Series
        Decimal odds (must be >= 1.0)
    
    Returns
    -------
    implied_prob : float, np.ndarray, or pd.Series
        Implied probability (between 0 and 1)
    
    Examples
    --------
    >>> implied_from_decimal(2.0)
    0.5
    >>> implied_from_decimal(1.91)
    0.5235602094240838
    """
    decimal = np.asarray(decimal_odds, dtype=float)
    
    # Validate
    if np.any(decimal < 1.0):
        raise ValueError("Decimal odds must be >= 1.0")
    
    implied = 1.0 / decimal
    
    # Return same type as input
    if isinstance(decimal_odds, pd.Series):
        return pd.Series(implied, index=decimal_odds.index)
    elif isinstance(decimal_odds, (int, float)):
        return float(implied)
    else:
        return implied


# ============================================================================
# DE-VIG FUNCTIONS
# ============================================================================

def proportional_devig(implied_probs: Union[List, np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Remove vig using proportional (multiplicative) method.
    
    This is the SIMPLEST de-vig method. It assumes the bookmaker applies
    the same percentage margin to all outcomes proportionally.
    
    Method:
        fair_prob[i] = implied_prob[i] / sum(implied_probs)
    
    Advantages:
        - Simple, transparent
        - Guaranteed to sum to 1.0
        - Fast computation
    
    Disadvantages:
        - Assumes equal vig across all outcomes (often false)
        - Can distort longshot probabilities
    
    Parameters
    ----------
    implied_probs : list, np.ndarray, or pd.Series
        Raw implied probabilities (should sum to > 1.0)
    
    Returns
    -------
    fair_probs : np.ndarray or pd.Series
        De-vigged probabilities (sum = 1.0)
    
    Examples
    --------
    >>> implied = [0.52, 0.48, 0.05]  # Sum = 1.05 (5% overround)
    >>> proportional_devig(implied)
    array([0.49523810, 0.45714286, 0.04761905])  # Sum = 1.0
    """
    probs = np.asarray(implied_probs, dtype=float)
    
    # Validate
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Implied probabilities must be between 0 and 1")
    
    total = probs.sum()
    
    if total <= 0:
        raise ValueError("Sum of implied probabilities must be > 0")
    
    # Normalize to sum to 1.0
    fair = probs / total
    
    # Return same type as input
    if isinstance(implied_probs, pd.Series):
        return pd.Series(fair, index=implied_probs.index)
    else:
        return fair


def power_devig(implied_probs: Union[List, np.ndarray, pd.Series], k: float = None) -> Union[np.ndarray, pd.Series]:
    """
    Remove vig using power method (exponential de-vig).
    
    This method applies a power transformation to adjust for the fact that
    bookmakers often apply higher margins to longshots than favorites.
    
    Method:
        1. Find k such that sum(prob[i]^k) = 1.0
        2. fair_prob[i] = prob[i]^k
    
    Advantages:
        - Better for markets with large favorite/longshot spreads
        - Reduces overround on longshots more than favorites
    
    Disadvantages:
        - Requires numerical optimization (slower)
        - Less transparent than proportional
    
    Parameters
    ----------
    implied_probs : list, np.ndarray, or pd.Series
        Raw implied probabilities
    k : float, optional
        Power parameter. If None, will be solved numerically.
        Typical range: 0.5 to 1.5
    
    Returns
    -------
    fair_probs : np.ndarray or pd.Series
        De-vigged probabilities (sum ≈ 1.0)
    
    Examples
    --------
    >>> implied = np.array([0.52, 0.48])  # Sum = 1.0 (already fair)
    >>> power_devig(implied)
    array([0.52, 0.48])  # No change needed
    """
    from scipy.optimize import fsolve
    
    probs = np.asarray(implied_probs, dtype=float)
    
    # Validate
    if np.any(probs <= 0) or np.any(probs >= 1):
        raise ValueError("Implied probabilities must be in (0, 1)")
    
    total = probs.sum()
    
    # If already fair, return as-is
    if np.isclose(total, 1.0, atol=1e-6):
        return probs if not isinstance(implied_probs, pd.Series) else pd.Series(probs, index=implied_probs.index)
    
    # Solve for k such that sum(prob^k) = 1.0
    if k is None:
        def objective(k):
            return np.sum(probs ** k) - 1.0
        
        # Initial guess: k slightly < 1.0 if overround, slightly > 1.0 if underround
        k_init = 0.9 if total > 1.0 else 1.1
        k_solution = fsolve(objective, k_init)[0]
    else:
        k_solution = k
    
    # Apply power transformation
    fair = probs ** k_solution
    
    # Normalize to exactly 1.0 (numerical precision)
    fair = fair / fair.sum()
    
    # Return same type as input
    if isinstance(implied_probs, pd.Series):
        return pd.Series(fair, index=implied_probs.index)
    else:
        return fair


def additive_devig(implied_probs: Union[List, np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Remove vig using additive (uniform) method.
    
    This method assumes the bookmaker adds the SAME absolute margin to each outcome.
    
    Method:
        overround = sum(implied_probs) - 1.0
        fair_prob[i] = implied_prob[i] - (overround / n)
    
    Advantages:
        - Simple concept
        - Can be better for markets with similar-probability outcomes
    
    Disadvantages:
        - Can produce negative probabilities for longshots (need to clip)
        - Less commonly used than proportional or power
    
    Parameters
    ----------
    implied_probs : list, np.ndarray, or pd.Series
        Raw implied probabilities
    
    Returns
    -------
    fair_probs : np.ndarray or pd.Series
        De-vigged probabilities (sum ≈ 1.0)
    
    Examples
    --------
    >>> implied = np.array([0.53, 0.52])  # Sum = 1.05 (5% overround)
    >>> additive_devig(implied)
    array([0.505, 0.495])  # Sum = 1.0
    """
    probs = np.asarray(implied_probs, dtype=float)
    
    # Validate
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Implied probabilities must be between 0 and 1")
    
    n = len(probs)
    total = probs.sum()
    overround = total - 1.0
    
    # Subtract equal amount from each outcome
    fair = probs - (overround / n)
    
    # Clip negative values (can happen with longshots)
    fair = np.clip(fair, 0.0, 1.0)
    
    # Renormalize to exactly 1.0
    fair = fair / fair.sum()
    
    # Return same type as input
    if isinstance(implied_probs, pd.Series):
        return pd.Series(fair, index=implied_probs.index)
    else:
        return fair


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_overround(implied_probs: Union[List, np.ndarray, pd.Series]) -> float:
    """
    Calculate overround (bookmaker margin) from implied probabilities.
    
    Overround = sum of all implied probabilities in a market.
    
    Interpretation:
        1.00 = fair market (no vig)
        1.05 = 5% overround (bookmaker has 5% margin)
        1.18 = 18% overround (typical PGA outrights market)
    
    Parameters
    ----------
    implied_probs : list, np.ndarray, or pd.Series
        Raw implied probabilities
    
    Returns
    -------
    overround : float
        Sum of implied probabilities (should be > 1.0)
    
    Examples
    --------
    >>> calculate_overround([0.52, 0.48, 0.05])
    1.05
    """
    return np.sum(implied_probs)


def calculate_margin_percentage(overround: float) -> float:
    """
    Convert overround to margin percentage.
    
    Margin % = (overround - 1.0) * 100
    
    Parameters
    ----------
    overround : float
        Overround value (typically > 1.0)
    
    Returns
    -------
    margin_pct : float
        Margin as percentage
    
    Examples
    --------
    >>> calculate_margin_percentage(1.18)
    18.0
    """
    return (overround - 1.0) * 100.0


def compare_devig_methods(implied_probs: Union[List, np.ndarray, pd.Series], 
                         labels: List[str] = None) -> pd.DataFrame:
    """
    Compare all three de-vig methods side-by-side.
    
    Useful for understanding how different methods affect probabilities,
    especially for favorites vs. longshots.
    
    Parameters
    ----------
    implied_probs : list, np.ndarray, or pd.Series
        Raw implied probabilities
    labels : list of str, optional
        Labels for each outcome (e.g., player names)
    
    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame with columns for each method
    
    Examples
    --------
    >>> implied = [0.55, 0.45, 0.10]
    >>> compare_devig_methods(implied, labels=['Favorite', 'Second', 'Longshot'])
    """
    probs = np.asarray(implied_probs, dtype=float)
    
    # Generate labels if not provided
    if labels is None:
        labels = [f"Outcome_{i+1}" for i in range(len(probs))]
    
    # Calculate all methods
    proportional = proportional_devig(probs)
    power = power_devig(probs)
    additive = additive_devig(probs)
    
    # Create comparison DataFrame
    df = pd.DataFrame({
        'Label': labels,
        'Raw Implied': probs,
        'Proportional': proportional,
        'Power': power,
        'Additive': additive
    })
    
    # Add summary row
    summary = pd.DataFrame({
        'Label': ['SUM'],
        'Raw Implied': [probs.sum()],
        'Proportional': [proportional.sum()],
        'Power': [power.sum()],
        'Additive': [additive.sum()]
    })
    
    df = pd.concat([df, summary], ignore_index=True)
    
    return df


# ============================================================================
# EXPECTED VALUE CALCULATION
# ============================================================================

def calculate_ev(fair_prob: Union[float, np.ndarray, pd.Series],
                 decimal_odds: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate expected value (EV) of a bet.
    
    EV = (fair_prob * profit) - ((1 - fair_prob) * stake)
    
    For a $1 stake:
        EV = fair_prob * (decimal_odds - 1) - (1 - fair_prob)
        EV = fair_prob * decimal_odds - 1
    
    Interpretation:
        EV > 0: Positive expected value (good bet)
        EV = 0: Fair bet (no edge)
        EV < 0: Negative expected value (bad bet)
    
    Parameters
    ----------
    fair_prob : float, np.ndarray, or pd.Series
        Your estimated fair probability
    decimal_odds : float, np.ndarray, or pd.Series
        Bookmaker's decimal odds
    
    Returns
    -------
    ev : float, np.ndarray, or pd.Series
        Expected value per $1 wagered
    
    Examples
    --------
    >>> calculate_ev(0.25, 5.0)  # 25% chance, 5.0 decimal odds
    0.25  # 25 cents profit per $1 bet (positive EV)
    >>> calculate_ev(0.20, 4.0)  # 20% chance, 4.0 decimal odds
    -0.2  # 20 cents loss per $1 bet (negative EV)
    """
    prob = np.asarray(fair_prob, dtype=float)
    odds = np.asarray(decimal_odds, dtype=float)
    
    # EV = P * (odds - 1) - (1 - P) = P * odds - 1
    ev = prob * odds - 1.0
    
    # Return same type as input
    if isinstance(fair_prob, pd.Series):
        return pd.Series(ev, index=fair_prob.index)
    elif isinstance(fair_prob, (int, float)):
        return float(ev)
    else:
        return ev


def calculate_edge(fair_prob: Union[float, np.ndarray, pd.Series],
                   market_prob: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate betting edge (difference between fair and market probability).
    
    Edge = fair_prob - market_prob
    
    Interpretation:
        Edge > 0: Market underpricing (good bet)
        Edge = 0: Fair price
        Edge < 0: Market overpricing (avoid)
    
    Parameters
    ----------
    fair_prob : float, np.ndarray, or pd.Series
        Your estimated fair probability
    market_prob : float, np.ndarray, or pd.Series
        Market-implied probability (after de-vig)
    
    Returns
    -------
    edge : float, np.ndarray, or pd.Series
        Probability difference
    
    Examples
    --------
    >>> calculate_edge(0.15, 0.10)  # You think 15%, market says 10%
    0.05  # 5 percentage point edge
    """
    fair = np.asarray(fair_prob, dtype=float)
    market = np.asarray(market_prob, dtype=float)
    
    edge = fair - market
    
    # Return same type as input
    if isinstance(fair_prob, pd.Series):
        return pd.Series(edge, index=fair_prob.index)
    elif isinstance(fair_prob, (int, float)):
        return float(edge)
    else:
        return edge


# ============================================================================
# UNIT TESTS (run with: python devig_utils.py)
# ============================================================================

if __name__ == "__main__":
    print("Running unit tests for devig_utils.py...\n")
    
    # Test 1: American to decimal conversion
    print("Test 1: American to Decimal Conversion")
    assert american_to_decimal(100) == 2.0, "Failed: +100 should be 2.0"
    assert np.isclose(american_to_decimal(-110), 1.909090909, atol=1e-6), "Failed: -110 conversion"
    assert american_to_decimal(250) == 3.5, "Failed: +250 should be 3.5"
    print("✓ PASSED\n")
    
    # Test 2: Implied probability
    print("Test 2: Implied Probability")
    assert implied_from_american(100) == 0.5, "Failed: +100 should be 50%"
    assert np.isclose(implied_from_american(-110), 0.5238, atol=0.0001), "Failed: -110 implied"
    print("✓ PASSED\n")
    
    # Test 3: Proportional de-vig
    print("Test 3: Proportional De-vig")
    implied = np.array([0.55, 0.55])  # 110% market (10% overround)
    devigged = proportional_devig(implied)
    assert np.isclose(devigged.sum(), 1.0, atol=1e-10), "Failed: Should sum to 1.0"
    assert np.isclose(devigged[0], 0.5, atol=1e-10), "Failed: Should be 50/50"
    print("✓ PASSED\n")
    
    # Test 4: Overround calculation
    print("Test 4: Overround Calculation")
    assert np.isclose(calculate_overround([0.52, 0.48, 0.05]), 1.05, atol=1e-10), "Failed: Should be 1.05"
    assert np.isclose(calculate_margin_percentage(1.18), 18.0, atol=1e-10), "Failed: Should be 18%"
    print("✓ PASSED\n")
    
    # Test 5: EV calculation
    print("Test 5: Expected Value")
    assert np.isclose(calculate_ev(0.25, 5.0), 0.25, atol=1e-10), "Failed: EV should be +0.25"
    assert np.isclose(calculate_ev(0.5, 2.0), 0.0, atol=1e-10), "Failed: Fair bet should be EV=0"
    print("✓ PASSED\n")
    
    # Test 6: Edge calculation
    print("Test 6: Edge Calculation")
    assert np.isclose(calculate_edge(0.15, 0.10), 0.05, atol=1e-10), "Failed: Edge should be 0.05"
    print("✓ PASSED\n")
    
    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)