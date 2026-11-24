# Changelog (`v1.0` ~ `v1.2.1`)

All notable changes to **`dual-layer-mad`** (1D Dual-Layer MAD core) are documented in this file.
*(For multidimensional `SpatialNDDLMAD`, `TabularNDDLMAD`, `TP-GGD-Fast`, `Hybrid*`, `RecursiveGroupDLMAD`, and 6-language `Cito/Fusion` SSOT, see the `nd-dlmad` private repository.)*

---

## [v1.2.1] - 2026-11-24 — Sub-LSB Lattice Perturbation Hardening

### Fixed
- Hardened `_excess_point_mass_list` and `_excess_point_mass_array` in `dual_layer_mad/core.py` against sub-LSB float perturbations (e.g. `1e-4` drift on `uint8` integer lattices) by using `ref_off = min(3, nz_cnt // 150)` and `2 * n1 >= first_gt` lattice classification.

---

## [v1.2.0] - 2026-11-24 — Zero-Inflation Double-Sawtooth Fix

### Fixed
- **Zero-Inflation "Double Sawtooth Collapse" (`_dual_mad_std_from_list`, `_dual_mad_std_from_array`)**:
  - Added Twin-Adjacent-Shell excess point-mass estimation (`_excess_point_mass_*`), `mad1` debiasing (`_debias_mad1_point_mass`), and Level-2 mixture-quantile Banach fixed-point solver (`_mixture_mad_fixed_point`).
  - Eliminated Tooth #1 (`p0 ~ 0.25-0.34` dip) and Tooth #2 (`p0 ~ 0.48-0.50` collapse / even-`N` `0.0000`) across `p0 = 0.00 ~ 0.60`, while keeping ordinary `uint8` quantization (`p0 = 0`) inflation-free and preserving the 50% global outlier breakdown point.
- Added `test_zero_inflation_sweep_quantization_and_breakdown` (`6/6` passed).

---

## [v1.1.0] - 2026-11-23 — Truncated-Normal Fixed-Point & Non-Zero Fallback

### Fixed
- Replaced static `1.4826` (`-10.46%` downward bias) and static `1.65568` (`+8~12%` contamination upward bias) with 4-step conditional truncated-normal Banach fixed-point correction (`_truncated_mad_fixed_point`).
- Fixed `get_dual_mad_noise` half-normal folding (`np.abs(np.diff(...))`) and missing `sqrt(2)` normalization.
- Added `mad1 <= tol` non-zero deviation fallback (`eff_mad1`) for sparse/quantized windows.

---

## [v1.0.0] - Initial Release
- Original 1D Dual-Layer MAD (`DualLayerMAD`, `get_dual_mad_noise`): Sorting -> Level-1 MAD outlier gate ($R_1 = 2.5\cdot\text{MAD}_1$) -> Level-2 MAD × `1.4826`.
