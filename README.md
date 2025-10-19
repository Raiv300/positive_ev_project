# Positive Expected Value in Golf Betting Markets

**Sprint 2 Complete: Exploratory Data Analysis & Problem Refinement**

**Author:** [Matt Raivel]  
**Course:** ADSP - Applied Data Science Project  
**Last Updated:** October 19, 2025

---

## Project Overview

This project investigates whether machine learning models can identify systematic mispricing in professional golf betting markets. Using data from the DataGolf API and historical betting odds, I analyze whether predictive models can detect when market-implied probabilities systematically diverge from true win probabilities, particularly in the mid-probability range (2-10%) where betting value opportunities may exist.

## Refined Research Question (Post-EDA)

**"Can machine learning models identify positive expected value opportunities by detecting when devigged market probabilities systematically misprice player performance in the 2-10% implied probability range?"**

This represents a pivot from the original goal of "predicting winners better than markets." EDA revealed that markets are highly efficient at extremes (favorites and long shots), but the mid-tier shows more variance. Additionally, the extreme class imbalance (1:150 winner ratio) makes direct classification intractable, so the focus shifted to probabilistic calibration and Expected Value analysis.

---

## Repository Structure

```
POSITIVE_EV_PROJECT/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── interim/                      # Intermediate datasets
│   │   ├── events.csv                # Tournament info
│   │   ├── final_dataset.csv         # Master merged dataset
│   │   ├── player_odds_data.csv      # Betting odds
│   │   └── rounds.csv                # Round scores
│   ├── processed/                    # ML-ready data
│   │   └── model_data_clean.csv
│   └── figures/                      # Visualizations
│
├── notebooks/
│   ├── 01_Cleaning.ipynb             # Data cleaning
│   ├── 01_DataCleaning.ipynb         # Enhanced cleaning
│   └── 02_EDA.ipynb                  # Exploratory analysis
│
└── src/Positive_EV_Repo/
    ├── data/
    │   ├── datagolfapi.py            # API wrapper
    │   ├── devig_utils.py            # Odds devigging
    │   ├── fetch_datagolf.py         # Data fetching
    │   └── testapi.py                # API testing
    └── models/
        ├── backtest.py               # Backtesting framework
        └── calibrate.py              # Probability calibration
```

---

## Sprint 2 Key Accomplishments

### Data Collection & Integration 
Built a complete data pipeline integrating DataGolf API, historical betting odds, and tournament results. Created custom odds devigging utilities to remove bookmaker margins and calculate fair value probabilities. Final dataset contains ~XX,XXX player-tournament observations from 2020-2025.

### Data Cleaning 
Developed comprehensive cleaning pipeline handling missing values through domain-informed imputation strategies. Missing course history filled with recent form averages, world rankings imputed using tournament-specific medians. Removed 247 duplicates and standardized 18 player name variants. Final cleaned dataset in `data/processed/model_data_clean.csv`.

### Exploratory Data Analysis 
Key findings from EDA revealed:
- **Market efficiency varies by probability range**: Favorites (>15%) and long shots (<1%) perform close to market expectations
- **Mid-tier opportunity**: Players priced 2-10% show higher variance than odds suggest
- **Extreme class imbalance**: 1:150 winner ratio requires specialized ML techniques  
- **Temporal stability**: Pre-tournament odds are remarkably stable with minimal line movement
- **Course-fit patterns**: Player archetypes (long hitters, accurate irons) show venue-specific performance not fully captured by markets

### Challenges Identified 
Severe class imbalance requires cost-sensitive learning and specialized metrics (AUPRC, log loss, Brier score). Limited granular strokes-gained data constrains course-fit modeling. Small sample sizes for subgroup analyses necessitate careful regularization. Market efficiency means any edge will be modest and require sophisticated feature engineering.

---

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/raiv300/POSITIVE_EV_PROJECT.git
cd POSITIVE_EV_PROJECT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify data
ls data/interim/
```

---

## Reproduction Instructions

**Run EDA Analysis:**
```bash
jupyter notebook notebooks/02_EDA.ipynb
# Execute all cells to reproduce visualizations and findings
```

**Load Cleaned Data:**
```python
import pandas as pd
df = pd.read_csv('data/processed/model_data_clean.csv')
```

---

## Sprint 3 Roadmap

### Modeling Approach
- **XGBoost** with cost-sensitive learning (`scale_pos_weight` for class imbalance)
- **LightGBM** for efficiency with large datasets
- **Calibrated Logistic Regression** as interpretable baseline
- **Neural Network (MLP)** with dropout and class weights

### Evaluation Metrics
- Log Loss (primary optimization metric)
- Brier Score with decomposition (calibration quality)
- AUPRC (handles imbalance better than ROC-AUC)
- Expected Value simulation (practical betting ROI)
- Calibration plots (reliability diagrams)

### Validation Strategy
- Time-based train/validation/test split (60/20/20)
- Grouped cross-validation by tournament
- Walk-forward validation for temporal robustness

---

## Key Technologies

- Python 3.10, pandas, NumPy, scikit-learn
- XGBoost/LightGBM for gradient boosting
- SHAP for model interpretability
- Matplotlib/Seaborn for visualization
- Jupyter notebooks for analysis

---

## Data Description

**Sources**: DataGolf API, historical betting odds, tournament results  
**Time Period**: 2020-2025  
**Observations**: ~XX,XXX player-tournament records  

**Key Variables**: player_name, event_id, decimal_odds, devigged_odds, implied_probability, finish_position, won_tournament, world_ranking, recent_form_avg, course_history

See `data/interim/final_dataset.csv` for complete master dataset.

---

## Contact

**Matt Raivel**  
Email: mraivel@terpmail.umd.edu
GitHub: [@raiv300](https://github.com/raiv300)

---

## License

Academic project completed as coursework. DataGolf API data used under their terms of service.

---

**Project Status:** ✅ Sprint 2 Complete - Ready for Modeling Phase