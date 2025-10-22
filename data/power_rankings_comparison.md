# NFL Power Rankings Comparison - Week 7, 2025

**Generated:** 2025-10-22 00:29:40

## Algorithm 1: R1+SOV (baseline)

**Methodology:** Record-first + strength of victory
- Win%: 60%, Playoff Prob: 15%, Seed: 10%, Strength of Victory: 15%

| Rank | Team | Record | Score | Win% | Playoff% | Seed | SOV |
|------|------|--------|-------|------|----------|------|-----|
|  1 | IND | 6-1   |  83.9 |  85.7 |  94.3 | 100.0 |  55.6 |
|  2 | GB  | 4-1-1 |  77.5 |  75.0 |  77.9 | 100.0 |  72.2 |
|  3 | PHI | 5-2   |  77.2 |  71.4 |  91.5 |  66.7 |  93.3 |
|  4 | DET | 5-2   |  76.0 |  71.4 |  88.6 |  93.3 |  70.4 |
|  5 | NE  | 5-2   |  73.1 |  71.4 |  87.7 |  93.3 |  51.9 |
|  6 | SF  | 5-2   |  73.0 |  71.4 |  78.7 |  73.3 |  73.3 |
|  7 | TB  | 5-2   |  72.7 |  71.4 |  88.9 |  60.0 |  70.4 |
|  8 | LAR | 5-2   |  72.7 |  71.4 |  76.4 |  86.7 |  64.4 |
|  9 | DEN | 5-2   |  71.0 |  71.4 |  81.1 |  86.7 |  48.9 |
| 10 | SEA | 5-2   |  69.5 |  71.4 |  61.9 |  80.0 |  62.2 |
| 11 | PIT | 4-2   |  66.0 |  66.7 |  66.3 |  73.3 |  58.3 |
| 12 | BUF | 4-2   |  63.2 |  66.7 |  83.5 |  80.0 |  17.6 |
| 13 | KC  | 4-3   |  61.7 |  57.1 |  81.8 |  66.7 |  56.5 |
| 14 | JAX | 4-3   |  60.1 |  57.1 |  46.8 |  60.0 |  85.2 |
| 15 | CHI | 4-2   |  58.3 |  66.7 |  33.9 |  53.3 |  52.8 |

---

## Algorithm 2: R1+SOV + Point Differential (experimental)

**Methodology:** Baseline R1+SOV with 10% point differential contribution
- Win%: 50%, Playoff Prob: 15%, Seed: 10%, SOV: 15%, Point Diff: 10%

| Rank | Team | Record | Score | Win% | Playoff% | Seed | SOV | Pt Diff |
|------|------|--------|-------|------|----------|------|-----|---------|
|  1 | IND | 6-1   |  87.1 |  85.7 |  94.3 | 100.0 |  55.6 |   100.0 |
|  2 | DET | 5-2   |  77.3 |  71.4 |  88.6 |  93.3 |  70.4 |    85.1 |
|  3 | GB  | 4-1-1 |  76.6 |  75.0 |  77.9 | 100.0 |  72.2 |    68.6 |
|  4 | LAR | 5-2   |  74.3 |  71.4 |  76.4 |  86.7 |  64.4 |    81.9 |
|  5 | NE  | 5-2   |  74.1 |  71.4 |  87.7 |  93.3 |  51.9 |    76.6 |
|  6 | PHI | 5-2   |  72.5 |  71.4 |  91.5 |  66.7 |  93.3 |    53.7 |
|  7 | SEA | 5-2   |  71.9 |  71.4 |  61.9 |  80.0 |  62.2 |    81.4 |
|  8 | DEN | 5-2   |  71.5 |  71.4 |  81.1 |  86.7 |  48.9 |    70.2 |
|  9 | SF  | 5-2   |  70.0 |  71.4 |  78.7 |  73.3 |  73.3 |    54.8 |
| 10 | TB  | 5-2   |  68.8 |  71.4 |  88.9 |  60.0 |  70.4 |    50.5 |
| 11 | PIT | 4-2   |  64.9 |  66.7 |  66.3 |  73.3 |  58.3 |    56.4 |
| 12 | BUF | 4-2   |  64.8 |  66.7 |  83.5 |  80.0 |  17.6 |    67.0 |
| 13 | KC  | 4-3   |  64.5 |  57.1 |  81.8 |  66.7 |  56.5 |    84.0 |
| 14 | CHI | 4-2   |  58.1 |  66.7 |  33.9 |  53.3 |  52.8 |    49.5 |
| 15 | JAX | 4-3   |  57.6 |  57.1 |  46.8 |  60.0 |  85.2 |    46.3 |

---

## Algorithm 3: Weighted Composite

**Methodology:** Weighted blend of multiple metrics
- Sagarin: 15%, Playoff Prob: 25%, Point Diff: 20%, Recent Form: 15%,
  Record Quality: 10%, EPA: 10%, Turnovers: 5%

| Rank | Team | Record | Score | Sagarin | Playoff% | Pt Diff | Form |
|------|------|--------|-------|---------|----------|---------|------|
|  1 | IND | 6-1   |  84.8 | 67.7 |  94.3 | 100.0 | 100.0 |
|  2 | DET | 5-2   |  78.8 | 100.0 |  88.6 |  85.1 |  66.7 |
|  3 | NE  | 5-2   |  75.5 | 57.7 |  87.7 |  76.6 | 100.0 |
|  4 | DEN | 5-2   |  75.2 | 74.8 |  81.1 |  70.2 | 100.0 |
|  5 | KC  | 4-3   |  75.0 | 96.9 |  81.8 |  84.0 |  66.7 |
|  6 | GB  | 4-1-1 |  73.7 | 86.7 |  77.9 |  68.6 |  83.3 |
|  7 | LAR | 5-2   |  72.0 | 79.2 |  76.4 |  81.9 |  66.7 |
|  8 | PHI | 5-2   |  67.7 | 96.0 |  91.5 |  53.7 |  33.3 |
|  9 | TB  | 5-2   |  67.3 | 68.6 |  88.9 |  50.5 |  66.7 |
| 10 | SEA | 5-2   |  66.6 | 68.1 |  61.9 |  81.4 |  66.7 |
| 11 | BUF | 4-2   |  65.4 | 79.4 |  83.5 |  67.0 |  33.3 |
| 12 | SF  | 5-2   |  64.5 | 61.4 |  78.7 |  54.8 |  66.7 |
| 13 | PIT | 4-2   |  61.0 | 59.6 |  66.3 |  56.4 |  66.7 |
| 14 | CHI | 4-2   |  56.0 | 56.5 |  33.9 |  49.5 | 100.0 |
| 15 | HOU | 2-4   |  54.9 | 68.3 |  37.8 |  71.8 |  66.7 |

---

## Algorithm 2: Weighted Composite - Low Sagarin (5%)

**Methodology:** Weighted blend with reduced Sagarin influence
- Sagarin: 5% (-10%), Playoff Prob: 25%, Point Diff: 25% (+5%), Recent Form: 20% (+5%),
  Record Quality: 10%, EPA: 10%, Turnovers: 5%

| Rank | Team | Record | Score | Sagarin | Playoff% | Pt Diff | Form |
|------|------|--------|-------|---------|----------|---------|------|
|  1 | IND | 6-1   |  88.0 | 67.7 |  94.3 | 100.0 | 100.0 |
|  2 | NE  | 5-2   |  78.6 | 57.7 |  87.7 |  76.6 | 100.0 |
|  3 | DET | 5-2   |  76.4 | 100.0 |  88.6 |  85.1 |  66.7 |
|  4 | DEN | 5-2   |  76.2 | 74.8 |  81.1 |  70.2 | 100.0 |
|  5 | KC  | 4-3   |  72.9 | 96.9 |  81.8 |  84.0 |  66.7 |
|  6 | GB  | 4-1-1 |  72.6 | 86.7 |  77.9 |  68.6 |  83.3 |
|  7 | LAR | 5-2   |  71.5 | 79.2 |  76.4 |  81.9 |  66.7 |
|  8 | SEA | 5-2   |  67.2 | 68.1 |  61.9 |  81.4 |  66.7 |
|  9 | TB  | 5-2   |  66.3 | 68.6 |  88.9 |  50.5 |  66.7 |
| 10 | SF  | 5-2   |  64.4 | 61.4 |  78.7 |  54.8 |  66.7 |
| 11 | BUF | 4-2   |  62.4 | 79.4 |  83.5 |  67.0 |  33.3 |
| 12 | PHI | 5-2   |  62.4 | 96.0 |  91.5 |  53.7 |  33.3 |
| 13 | PIT | 4-2   |  61.2 | 59.6 |  66.3 |  56.4 |  66.7 |
| 14 | CHI | 4-2   |  57.8 | 56.5 |  33.9 |  49.5 | 100.0 |
| 15 | HOU | 2-4   |  55.0 | 68.3 |  37.8 |  71.8 |  66.7 |

---

## Algorithm 3: Weighted Composite - No Sagarin (0%)

**Methodology:** Weighted blend without Sagarin ratings
- Sagarin: 0%, Playoff Prob: 30% (+5%), Point Diff: 25% (+5%), Recent Form: 20% (+5%),
  Record Quality: 10%, EPA: 10%, Turnovers: 5%

| Rank | Team | Record | Score | Playoff% | Pt Diff | Form | EPA |
|------|------|--------|-------|----------|---------|------|-----|
|  1 | IND | 6-1   |  89.4 |  94.3 | 100.0 | 100.0 | 50.0 |
|  2 | NE  | 5-2   |  80.1 |  87.7 |  76.6 | 100.0 | 50.0 |
|  3 | DEN | 5-2   |  76.5 |  81.1 |  70.2 | 100.0 | 50.0 |
|  4 | DET | 5-2   |  75.8 |  88.6 |  85.1 |  66.7 | 50.0 |
|  5 | GB  | 4-1-1 |  72.2 |  77.9 |  68.6 |  83.3 | 50.0 |
|  6 | KC  | 4-3   |  72.1 |  81.8 |  84.0 |  66.7 | 50.0 |
|  7 | LAR | 5-2   |  71.4 |  76.4 |  81.9 |  66.7 | 50.0 |
|  8 | TB  | 5-2   |  67.3 |  88.9 |  50.5 |  66.7 | 50.0 |
|  9 | SEA | 5-2   |  66.9 |  61.9 |  81.4 |  66.7 | 50.0 |
| 10 | SF  | 5-2   |  65.3 |  78.7 |  54.8 |  66.7 | 50.0 |
| 11 | BUF | 4-2   |  62.6 |  83.5 |  67.0 |  33.3 | 50.0 |
| 12 | PHI | 5-2   |  62.2 |  91.5 |  53.7 |  33.3 | 50.0 |
| 13 | PIT | 4-2   |  61.5 |  66.3 |  56.4 |  66.7 | 50.0 |
| 14 | CHI | 4-2   |  56.7 |  33.9 |  49.5 | 100.0 | 50.0 |
| 15 | HOU | 2-4   |  53.5 |  37.8 |  71.8 |  66.7 | 50.0 |

---

## Algorithm 6: Elo Dynamic

**Methodology:** Elo rating system updated after each game
- K-factor: 25, Home advantage: 3 points, Margin of victory multiplier

| Rank | Team | Record | Elo Rating | Change from Start |
|------|------|--------|------------|-------------------|
|  1 | IND | 6-1   |  1625.3 |      +125.3 |
|  2 | LAR | 5-2   |  1603.9 |      +103.9 |
|  3 | SEA | 5-2   |  1590.0 |       +90.0 |
|  4 | DET | 5-2   |  1587.4 |       +87.4 |
|  5 | NE  | 5-2   |  1577.8 |       +77.8 |
|  6 | KC  | 4-3   |  1573.0 |       +73.0 |
|  7 | DEN | 5-2   |  1567.2 |       +67.2 |
|  8 | GB  | 4-1-1 |  1555.1 |       +55.1 |
|  9 | SF  | 5-2   |  1542.5 |       +42.5 |
| 10 | TB  | 5-2   |  1533.9 |       +33.9 |
| 11 | PHI | 5-2   |  1532.5 |       +32.5 |
| 12 | HOU | 2-4   |  1532.5 |       +32.5 |
| 13 | BUF | 4-2   |  1532.1 |       +32.1 |
| 14 | DAL | 3-3-1 |  1521.6 |       +21.6 |
| 15 | CHI | 4-2   |  1519.7 |       +19.7 |

---

## Algorithm 7: Multi-Factor Score

**Methodology:** Independent scoring across 6 dimensions (averaged)
- Record, Performance, Efficiency, Momentum, Predictive, Market

| Rank | Team | Record | Score | Record | Perform | Effic | Moment | Predict | Market |
|------|------|--------|-------|--------|---------|-------|--------|---------|--------|
|  1 | IND | 6-1   |  81.7 |  85.7 |  100.0 |  50.0 |  100.0 |    94.3 |   67.7 |
|  2 | DET | 5-2   |  77.5 |  71.4 |   85.1 |  50.0 |   66.7 |    88.6 |  100.0 |
|  3 | GB  | 4-1-1 |  69.9 |  75.0 |   68.6 |  50.0 |   83.3 |    77.9 |   86.7 |
|  4 | KC  | 4-3   |  69.9 |  57.1 |   84.0 |  50.0 |   66.7 |    81.8 |   96.9 |
|  5 | NE  | 5-2   |  69.7 |  71.4 |   76.6 |  50.0 |  100.0 |    87.7 |   57.7 |
|  6 | LAR | 5-2   |  68.5 |  71.4 |   81.9 |  50.0 |   66.7 |    76.4 |   79.2 |
|  7 | DEN | 5-2   |  68.2 |  71.4 |   70.2 |  50.0 |  100.0 |    81.1 |   74.8 |
|  8 | PHI | 5-2   |  68.0 |  71.4 |   53.7 |  50.0 |   33.3 |    91.5 |   96.0 |
|  9 | SEA | 5-2   |  66.6 |  71.4 |   81.4 |  50.0 |   66.7 |    61.9 |   68.1 |
| 10 | TB  | 5-2   |  65.5 |  71.4 |   50.5 |  50.0 |   66.7 |    88.9 |   68.6 |
| 11 | SF  | 5-2   |  61.5 |  71.4 |   54.8 |  50.0 |   66.7 |    78.7 |   61.4 |
| 12 | PIT | 4-2   |  58.0 |  66.7 |   56.4 |  50.0 |   66.7 |    66.3 |   59.6 |
| 13 | CHI | 4-2   |  56.3 |  66.7 |   49.5 |  50.0 |  100.0 |    33.9 |   56.5 |
| 14 | BUF | 4-2   |  56.2 |  66.7 |   67.0 |  50.0 |   33.3 |    83.5 |   79.4 |
| 15 | DAL | 3-3-1 |  51.1 |  50.0 |   59.6 |  50.0 |   66.7 |     8.0 |   45.2 |

---

## Algorithm 6: AI-Enhanced Rankings

**Methodology:** Weighted Composite + AI adjustments (±3 spots max)
- AI Model: Sonnet 3.7

| Rank | Team | Record | Base | AI Adj | Reasoning |
|------|------|--------|------|--------|-----------|
|  1 | IND | 6-1   |    1 |          | No adjustment |
|  2 | DET | 5-2   |    2 |          | No adjustment |
|  3 | NE  | 5-2   |    3 |          | No adjustment |
|  4 | KC  | 4-3   |    5 | (5→4)    | Reigning Super Bowl champions with excellent Sagar |
|  5 | DEN | 5-2   |    4 | (4→5)    | No adjustment |
|  6 | GB  | 4-1-1 |    6 |          | No adjustment |
|  7 | PHI | 5-2   |    8 | (8→7)    | Eagles have the second-highest Sagarin score (96.0 |
|  8 | LAR | 5-2   |    7 | (7→8)    | No adjustment |
|  9 | TB  | 5-2   |    9 |          | No adjustment |
| 10 | SEA | 5-2   |   10 |          | No adjustment |
| 11 | BUF | 4-2   |   11 |          | No adjustment |
| 12 | SF  | 5-2   |   12 |          | No adjustment |
| 13 | PIT | 4-2   |   13 |          | No adjustment |
| 14 | CHI | 4-2   |   14 |          | No adjustment |
| 15 | LAC | 4-3   |   16 | (16→15)  | No adjustment |

**AI Analysis:** After reviewing the NFL power rankings, I've made a few adjustments based on qualitative factors. I moved Kansas City up to #3 due to their championship pedigree and strong Sagarin rating. Philadelphia moves up to #6 given their elite Sagarin score and high playoff probability. Houston drops to #17 despite good metrics because their 2-4 record outweighs their statistical profile. Washington deserves to be higher at #18 with their strong point differential and impressive Sagarin score despite their losing record.


---

## Side-by-Side Comparison (Top 20)

| Team | R1 | R1+PD | Comp | Low-Sag | No-Sag | Elo | Multi | AI | Variance | Record |
|------|----|-------|------|---------|--------|-----|-------|-----|----------|--------|
| IND |  1 |     1 |    1 |       1 |      1 |   1 |     1 |     1 |        0 | 6-1    |
| DET |  4 |     2 |    2 |       3 |      4 |   4 |     2 |     2 |        2 | 5-2    |
| NE  |  5 |     5 |    3 |       2 |      2 |   5 |     5 |     3 |        3 | 5-2    |
| GB  |  2 |     3 |    6 |       6 |      5 |   8 |     3 |     6 |        6 | 4-1-1  |
| DEN |  9 |     8 |    4 |       4 |      3 |   7 |     7 |     5 |        6 | 5-2    |
| LAR |  8 |     4 |    7 |       7 |      7 |   2 |     6 |     8 |        6 | 5-2    |
| KC  | 13 |    13 |    5 |       5 |      6 |   6 |     4 |     4 |        9 | 4-3    |
| SEA | 10 |     7 |   10 |       8 |      9 |   3 |     9 |    10 |        7 | 5-2    |
| PHI |  3 |     6 |    8 |      12 |     12 |  11 |     8 |     7 |        9 | 5-2    |
| TB  |  7 |    10 |    9 |       9 |      8 |  10 |    10 |     9 |        3 | 5-2    |
| SF  |  6 |     9 |   12 |      10 |     10 |   9 |    11 |    12 |        6 | 5-2    |
| BUF | 12 |    12 |   11 |      11 |     11 |  13 |    14 |    11 |        3 | 4-2    |
| PIT | 11 |    11 |   13 |      13 |     13 |  16 |    12 |    13 |        5 | 4-2    |
| CHI | 15 |    14 |   14 |      14 |     14 |  15 |    13 |    14 |        2 | 4-2    |
| CAR | 18 |    17 |   17 |      16 |     16 |  18 |    19 |    17 |        3 | 4-3    |
| LAC | 16 |    16 |   16 |      18 |     18 |  21 |    18 |    15 |        6 | 4-3    |
| HOU | 23 |    23 |   15 |      15 |     15 |  12 |    22 |    16 |       11 | 2-4    |
| JAX | 14 |    15 |   19 |      19 |     19 |  22 |    16 |    20 |        8 | 4-3    |
| ATL | 17 |    19 |   18 |      17 |     17 |  19 |    20 |    18 |        3 | 3-3    |
| DAL | 21 |    20 |   22 |      20 |     20 |  14 |    15 |    22 |        8 | 3-3-1  |

---

## Algorithm Correlation Matrix

**Spearman Correlation** (1.0 = perfect agreement)

| Comparison | Correlation |
|------------|-------------|
| R1 vs R1+PointDiff             |       0.984 |
| R1 vs Composite (15% Sag)      |       0.940 |
| R1 vs Elo                      |       0.882 |
| R1 vs Multi-Factor             |       0.948 |
| Composite (15% Sag) vs Low Sag (5%) |       0.990 |
| Composite (15% Sag) vs No Sag (0%) |       0.989 |
| Low Sag (5%) vs No Sag (0%)    |       0.998 |
| Composite vs Elo               |       0.943 |
| Composite vs Multi-Factor      |       0.966 |
| No Sagarin vs Elo              |       0.956 |
| No Sagarin vs Multi-Factor     |       0.961 |
| R1 vs AI                       |       0.940 |
| Composite vs AI                |       1.000 |
| No Sagarin vs AI               |       0.989 |

---

## Key Findings

### Biggest Disagreements (Variance > 8)

- **KC**: Ranks range from 4 to 13 (variance: 9)
- **PHI**: Ranks range from 3 to 12 (variance: 9)
- **HOU**: Ranks range from 12 to 23 (variance: 11)

### Indianapolis Colts (5-1, #1 AFC Seed) Placement

- R1+SOV (baseline): **#1**
- R1+SOV + PointDiff: **#1**
- Composite (15% Sagarin): **#1**
- Low Sagarin (5%): **#1**
- No Sagarin (0%): **#1**
- Elo: **#1**
- Multi-Factor: **#1**
- AI-Enhanced: **#1**

Average rank: **#1.0**
Sagarin rank: **#1** (using Composite as reference)

---

## Recommendations

✅ **High Agreement** (avg correlation: 0.963)
- All algorithms show strong agreement, suggesting robust rankings
- Any algorithm would be reliable for production use

### Algorithm-Specific Notes

**Weighted Composite:**
- Best for: Balanced view emphasizing playoff probability and point differential
- Strengths: Highly customizable weights, captures multiple dimensions

**Elo Dynamic:**
- Best for: Game-by-game performance tracking, recency bias
- Strengths: Self-updating, mathematically principled, no arbitrary weights

**Multi-Factor:**
- Best for: Comprehensive evaluation across independent dimensions
- Strengths: Considers record, performance, efficiency, momentum, prediction, market

**AI-Enhanced:**
- Best for: Incorporating subjective 'eye test' and recent context
- Strengths: Adds human-like judgment, adapts to injuries and trends
