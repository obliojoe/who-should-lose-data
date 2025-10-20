# Impact 2.0 – Final Implementation Spec

### Overview

Switch to a **single utility-based impact score** that represents how much a game matters to a team given its current competitive context (fighting for playoffs, chasing division, etc.).
Use that same utility to determine both:

* which outcome helps/hurts the team (`root_against`), and
* how “big” the game is (`impact`).

This fixes cancellation cases and allows late-season fans to see exactly which games keep their team alive, even when odds are long.

---

## 1. Core Computation

For each **team T** and **unplayed game G**, run both outcomes (home wins / away wins).

From simulations you already have for T:

```
P_playoff, P_division, P_top_seed, expected_seed_value (ESV)
```

Compute signed differences (away − home):

```
d_playoff = P_playoff_A - P_playoff_H       # pp
d_div     = P_division_A - P_division_H     # pp
d_top     = P_top_seed_A - P_top_seed_H     # pp
d_seed    = ESV_A - ESV_H                   # 0–100 scale, can be treated as pp
```

---

## 2. Dynamic Weight Matrix

Choose weights based on team status.
We trust clinch/elimination info, so just check those flags:

| Condition                                     | playoff | division | top-seed | seed |
| --------------------------------------------- | ------- | -------- | -------- | ---- |
| **Playoffs not clinched/eliminated**          | 0.50    | 0.25     | 0.15     | 0.10 |
| **Playoffs clinched, division open**          | 0.10    | 0.45     | 0.30     | 0.15 |
| **Division clinched, #1 seed live**           | 0.10    | 0.05     | 0.55     | 0.30 |
| **All goals locked**                          | 0.05    | 0.05     | 0.20     | 0.70 |
| **Eliminated but still alive mathematically** | 0.70    | 0.20     | 0.05     | 0.05 |

(The last case emphasizes *staying alive* even at long odds.)

Any goal that’s actually impossible (eliminated) or clinched gets weight = 0.

---

## 3. Utility Score

```
utility_signed = (w_playoff * d_playoff)
               + (w_div     * d_div)
               + (w_top     * d_top)
               + (w_seed    * d_seed)
impact = abs(utility_signed)
root_against = outcome with negative utility_signed  # hurts T the most
```

---

## 4. Conflict & Explanation

Compute weighted component contributions:

```
contribs = {playoff: wP*d_playoff, division: wD*d_div, top_seed: wT*d_top, seed: wS*d_seed}
```

Then:

* `dominant_reason = key with largest |contrib|`
* `conflict = true` if some contribs > 0 and others < 0 (goals disagree)

---

## 5. Data Storage

Extend your current game object minimally:

```json
{
  "impact": 4.8,
  "root_against": "MIN",
  "impact_breakdown": {
    "signed_swings": {
      "playoff_pp": -1.2,
      "division_pp": 4.9,
      "top_seed_pp": -0.8,
      "seed_value_pp": 0.1
    },
    "weighted_contributions": {
      "playoff": -0.6,
      "division": 2.2,
      "top_seed": -0.1,
      "seed": 0.1
    },
    "dominant_reason": "division",
    "conflict": true
  },
  "utility_weights": {
    "playoff": 0.5,
    "division": 0.25,
    "top_seed": 0.15,
    "seed": 0.1
  },
  "clinch_state": {
    "playoffs": "open",
    "division": "open",
    "top_seed": "open"
  }
}
```

---

## 6. Implementation Notes

* **Consistent Units:** everything stored in percentage-point equivalents for readability.
* **Noise floor:** drop impact < 0.3 pp to avoid clutter.
* **Elimination case:** when T is almost dead, high playoff weights will surface “miracle chain” games that still mathematically extend survival.
* **Display:**

  * `impact` → main numeric rank
  * `dominant_reason` → short label (“Division swing ±4.9 pp”)
  * `conflict` → ⚠️ indicator (“Root split: top-seed vs division”)

---

## 7. Why This Works

* One logic path drives both rooting and ranking.
* No “max-swing” distortion; scores scale with true team priorities.
* Late-season clarity: eliminated/near-eliminated teams still see survival paths.
* Easy to tune later: just tweak four numbers per regime.

---

In short: **replace “max swing” with weighted utility using the table above, store contributions and conflicts, and use that utility for both ranking and rooting.** It’s deterministic, fan-intuitive, and trivial to evolve once you have live data.
