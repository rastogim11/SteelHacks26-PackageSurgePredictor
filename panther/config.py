"""
Panther Post - operational assumptions.

EVERY number in this file that is not measured from the data is marked
PLACEHOLDER. When Operations supplies real figures, change them here and
nowhere else.

Outstanding asks (from the email thread, unanswered as of 2026-09-20):
  1. Locker / storage counts per mailroom
  2. Current headcount + scheduled hours per mailroom
  3. Whether WTS can export the location field separately from the
     scanning-user field
"""
from __future__ import annotations

# --------------------------------------------------------------- paths
DATA_RAW      = 'packagestats.csv'
DATA_CLEANED  = 'files/packagestats_cleaned.xlsx'
OUTPUT_DIR    = 'outputs'

# ------------------------------------------------------- site registry
# Permanently closed - excluded from all modelling.
EXCLUDE = ['Forbes', 'Darragh']

# WTS operator/user accounts that leak into the Mailroom column because the
# report conflates location with scanning user. These are NOT locations.
# Confirmed: the 'Gianni' rows are dated 2026-09-03..09 == the window of the
# team's own temp admin access, i.e. test scans.
OPERATOR_ACCOUNTS = [
    'Jessica Yacko', 'Devon Pleasant', 'Ed Youngman',
    'Gianni', 'Oscar Daniels',
]
# Non-residential / transhipment codes, also not mailrooms.
NON_MAILROOM_CODES = [
    'BSTWR3', 'Ship2Pitt1', 'Ship2Pitt4', 'Ship2Pitt5', 'Thomas Blvd',
]

LARGE         = ['Tower B', 'Sutherland']
OPEN_ALL_YEAR = ['Tower B', 'Residences on Bigelow']
ANCHOR        = 'Tower B'

# ------------------------------------------------------------- capacity
# PLACEHOLDER. Carried forward from panther_post_model.py v6, which split an
# assumed 1000-unit pool by historical volume share. The 1000 figure was never
# confirmed by Operations - it is inconsistent with observed occupancy, which
# exceeds 1000 system-wide on 432 of 731 days. Treat as a stand-in only.
TOTAL_LOCKERS = 1000
LARGE_SHARE   = 0.70

# Set to a dict once real counts arrive, e.g. {'Tower B': 1200, ...}.
# While None, lockers are derived by volume share exactly as v6 did.
LOCKERS_OVERRIDE: dict[str, int] | None = None

# ------------------------------------------------------------- staffing
# PLACEHOLDER. Headcount was requested in the email thread and never supplied.
# Values below are a stand-in so the pipeline runs end to end; they are NOT
# observed. Expressed as FTE currently assigned per mailroom.
HEADCOUNT_FTE: dict[str, float] = {
    'Tower B':               4.0,
    'Sutherland':            1.5,
    'Bouquet Gardens':       1.0,
    'Lothrop':               1.0,
    'Nordenberg':            1.0,
    'Residences on Bigelow': 0.5,
    'Ruskin':                0.5,
}
SHIFT_HOURS_PER_FTE = 8.0        # PLACEHOLDER
PRODUCTIVE_FRACTION = 0.75       # PLACEHOLDER: share of a shift on package work

# A mailroom that is open needs someone behind the counter regardless of
# parcel volume, so labour-hour arithmetic alone under-staffs quiet sites:
# Nordenberg computes to 0.5 FTE, and half a person cannot cover a counter.
# This floor is applied after the volume calculation.
MIN_COVERAGE_FTE = 1.0           # PLACEHOLDER pending opening hours from Ops

# WHAT THE LABOUR MODEL DOES NOT SEE.
# The WTS export covers parcel transactions only. The email thread refers to
# 'total mail received at mailroom' as a SEPARATE figure that was never
# supplied, and no data here covers letter mail, desk cover, key or equipment
# handling, returns paperwork beyond disposal, or cleaning. Required-FTE is
# therefore a FLOOR on parcel work, never a total establishment figure, and
# must never be read as evidence that a site is overstaffed.
LABOUR_SCOPE_NOTE = ('parcel transactions only; excludes letter mail, desk '
                     'cover, and non-parcel duties')

# Pay week runs Sunday..Saturday. Inferred from the timecard exchange in the
# email thread (period '9/6-9/12'; 2026-09-06 is a Sunday). All staffing
# outputs are aggregated on this boundary so they drop straight into the
# existing scheduling cadence.
WEEK_ANCHOR_DOW = 6              # Sunday, in pandas dayofweek terms

# --------------------------------------------------- measured constants
# MEASURED from consecutive scan timestamps within a burst (gap < 300s).
# Median per-package intake handling time, seconds. p75 is roughly double
# these and is what PLANNING_* uses, since the median assumes uninterrupted
# flow with no interruptions, walk time, or customer contact.
INTAKE_SECONDS_MEDIAN: dict[str, float] = {
    'Tower B':               37.0,
    'Sutherland':            40.0,
    'Nordenberg':            41.0,
    'Lothrop':               42.0,
    'Ruskin':                42.0,
    'Bouquet Gardens':       43.0,
    'Residences on Bigelow': 44.0,
}
INTAKE_SECONDS_DEFAULT = 41.0
PLANNING_MULTIPLIER    = 2.0     # median -> p75-ish planning rate

# Handout handling time. PLACEHOLDER pending the same burst analysis on the
# Delivered timestamps; a counter interaction is slower than a batch scan.
HANDOUT_SECONDS_DEFAULT = 75.0

# ------------------------------------------------------ model settings
SHORT_HORIZON   = 14             # days; short/long model split
EVENT_PAD       = 7
CV_FOLDS        = 5

# Candidates whose cross-validation score is within this fraction of the best
# are treated as tied, and the simplest of them is selected. Guards against
# picking a high-capacity model on a margin smaller than the noise. See the
# parsimony tie-break in selection.select_for_site.
SELECTION_TOLERANCE = 0.05
MIN_SITE_DAYS   = 150

# ------------------------------------------------------- train/test split
# Year one trains, year two tests. Data runs 2024-09-19 to 2026-09-19, so the
# cutoff falls exactly between the two academic years.
#
# Chosen over a percentage split because it is interpretable end to end: "the
# model saw last year and predicted this year." It costs something real - half
# the training data, and only ONE observation of each holiday in training
# instead of two - so scores will read lower than an 80/20 split would. They
# are also more honest, and a full year of test days covers every season,
# every break, and every event rather than only March to September.
SPLIT_DATE = '2025-09-19'
MIN_TRAIN_DAYS = 100             # year-one open days needed to fit a site

# Staff to a high quantile, not the mean: under-staffing costs queues and
# overtime, over-staffing costs mild idle. This is the operating quantile for
# staffing recommendations and the quantile whose pinball loss drives model
# selection.
STAFFING_QUANTILE = 0.80
REPORT_QUANTILES  = (0.05, 0.50, 0.80, 0.95)

# ------------------------------------------------------- dwell regimes
# Dwell behaviour at Tower B changed sharply between academic years. Share of
# arrivals still held past 90 days:
#     2024-25   0.9%        2025-26   6.7%
# No other site moved (all stay at or below 0.5% in both years), so this is a
# site-level operational change, not a campus trend. It drove Tower B's
# occupancy from roughly 500 to 2,690 parcels, and the backlog was only
# released by the 2026-07-01 purge of 1,806 items at a median age of 210 days.
#
# A single pooled survival curve cannot represent both regimes, so curves are
# stratified by regime. Boundaries are academic years starting 1 August.
REGIME_BOUNDARIES = [
    ('2024-25', '2024-08-01', '2025-07-31'),
    ('2025-26', '2025-08-01', '2026-07-31'),
    ('2026-27', '2026-08-01', '2027-07-31'),
]
# Which regime's dwell behaviour to assume when forecasting forward. The
# current partial year cannot yet exhibit long dwell - the export ends
# 2026-09-19, so a parcel received in August 2026 physically cannot show a
# 90-day hold - which makes its curves unusable for the tail.
FORECAST_REGIME = '2025-26'

# Administrative purge detection.
#
# A first attempt keyed on volume (pickups > 5x trailing median) failed: it
# flagged every post-break resumption spike, because the trailing median is
# depressed by the break itself. That censored 56% of Tower B's move-in
# stratum, which is exactly the legitimate student behaviour the model needs.
#
# The reliable signal is the AGE of what gets cleared, not the count:
#   2026-07-01  1806 items, median age 210 days, 99% over 30d  -> purge
#   2026-08-15   160 items, median age 248 days, 96% over 30d  -> purge
#   2026-01-13   419 items, median age  35 days                -> winter
#                backlog, collected by returning students, NOT a purge
#   2025-09-05   587 items, median age 0.2 days                -> move-in rush
# Normal high-volume days sit at a median age under 1 day.
PURGE_MIN_ITEMS       = 50    # bulk operation, not a busy afternoon
PURGE_MEDIAN_AGE_DAYS = 60    # above winter-backlog range (~27-45d), below
                              # observed purges (62-248d)