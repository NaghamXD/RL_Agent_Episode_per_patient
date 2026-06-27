# Reward-shaping ablation: results

Compares the 3 opt-in reward-shaping changes in `src/env/reward.py`
(see `scripts/sweep_reward_shaping.py`), layered in incrementally on top
of the winning hyperparameter regime (`sweep2_heavy_reg`), against
`control` (that regime's existing, already-evaluated result). All 5 runs
resume from the same shared `warmstart.pt`; each is evaluated on the same
40-patient validation split via `evaluate.py`.

Raw data: `reports/reward_shaping_summary.csv`. Validation-DVH curves:
`reports/reward_shaping_comparison.png`.

## Results

| variant | DVH | best ep | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem(54) | SpinalCord(45) | Mandible(70) | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|---|---|
| **control (sweep2, no changes)** | **23.34** | 1699 | 43.97 | 39.31 | 38.92 | 19.65 | 24.71 | 41.07 | 39.32 | 43.21 |
| C (soft coverage) | 26.82 | 500 | 33.76 | 28.08 | 29.00 | 10.09 | 21.03 | 29.53 | 31.12 | 31.44 |
| C_A (+ quadratic gap, lambda_phi unchanged) | 32.08 | 500 | 27.70 | 23.16 | 23.00 | 9.39 | 15.23 | 22.66 | 20.70 | 24.41 |
| C_A_retuned (+ lambda_phi=2.0) | 26.82 | 650 | 33.09 | 29.33 | 28.33 | 11.93 | 20.29 | 28.52 | 30.46 | 33.27 |
| C_A_B (+ soft OAR barrier, steepness=2.0) | 25.04 | 925 | 35.29 | 29.54 | 32.86 | 15.84 | 29.04 | 34.72 | 40.14 | 39.08 |

## Verdict: none of the 4 variants beat control

All four score worse than `control` on the aggregate DVH metric. The *why*
is informative enough to record:

- **C alone trades PTV coverage for OAR sparing, too aggressively.**
  Smoothing the hard coverage step gives partial credit for near-miss
  voxels, weakening the all-or-nothing pressure to fully reach Rx.
  D95_PTV70 drops 43.97 -> 33.76 in exchange for better OAR doses across
  every organ -- a real loss, not a wash.
- **C_A (unretuned) reproduces exactly the predicted magnitude-shrinkage
  problem.** Its validation-DVH curve never converges (oscillates
  32-51 the whole run, see the plot) -- squaring the gap without
  compensating `lambda_phi` left the PTV-shaping signal too weak relative
  to `lambda_oar`.
- **C_A_retuned validates the fix but doesn't improve beyond C.**
  `lambda_phi=2.0` recovers a stable curve similar to C alone, but lands
  at basically the same DVH/trade-off, not a further improvement.
- **C_A_B is the most informative result.** It's the only variant still
  improving when early stopping cut the others off (best at episode 925,
  still trending down) -- but it recovers PTV coverage at the cost of a
  *worse* SpinalCord dose than control (29.04 vs 24.71), and the parotids
  are still just as far over their 26 Gy tolerance as control (40.14/39.08
  vs 39.32/43.21). B's soft barrier didn't fix the parotid-overdose
  problem it was meant to address -- consistent with a flagged-but-untested
  caveat: B raises the OAR penalty's effective gain without `lambda_oar`
  being retuned down to compensate, the same kind of mismatch A had with
  `lambda_phi`.

## Recommendation

Keep the current reward formulation (`control`) for now -- none of these
changes are ready to adopt as-is. The most promising lead for future work
is a `C_A_B` variant with `lambda_oar` also retuned (not just
`lambda_phi`), since C_A_B was still improving when its run ended.
