"""Shared constants for the IEEE 14-bus SG-GFL tutorial."""

from pathlib import Path

import numpy as np


# Project paths and ANDES case.
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CASE_NAME = "ieee14/ieee14_regcp1.xlsx"
SYSTEM_BASE_MVA = 100.0

# Small-signal stability convention.
STABILITY_MARGIN = 1e-4

# Switchable synchronous generators in feature order: buses 2, 3 and 6.
SWITCHABLE_SGS = (
    {"bus": 2, "pv": 2, "generator": "GENROU_2", "governor": "TGOV1_2"},
    {"bus": 3, "pv": 3, "generator": "GENROU_3", "governor": "TGOV1_3"},
    {"bus": 6, "pv": 4, "generator": "GENROU_4", "governor": "TGOV1_4"},
)

# Bus 6 needs at least 30 MW because TGOV1_4.VMIN is 0.30 pu. The
# other two generators use the 0.10-pu PV minimum; all three have 0.50-pu Pmax.
SG_POWER_MIN_MW = (10.0, 10.0, 30.0)
SG_POWER_MAX_MW = (50.0, 50.0, 50.0)

# Grid-following converter and Slack limits.
GFL_PV_INDEX = 5  # Converter at bus 8 is the fifth row of the PV table.
GFL_POWER_MAX_MW = 100.0  # Consistent with the 100-MVA REGCP1 rating.

# Slack limits come from the original workbook and support pre-screening.
SLACK_POWER_MIN_MW = 50.0
SLACK_POWER_MAX_MW = 200.0

# Weak-grid modification used by every tutorial sample.
WEAK_GRID_LINE_INDEX = "Line_20"
WEAK_GRID_LINE_X = 0.17615 * 4.5  # Original reactance multiplied by 4.5.

# Load data and sampling ranges.
LOAD_BUSES = (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)
BASE_LOAD_POWER_MW = np.array(
    [21.7, 50.0, 47.8, 7.6, 15.0, 29.5, 9.0, 3.5, 6.1, 13.5, 20.0]
)
LOAD_TOTAL_SCALE_MIN = 0.80
LOAD_TOTAL_SCALE_MAX = 1.40
TARGET_SOLAR_MIN_MW = 92.0
TARGET_SOLAR_MAX_MW = 96.0
