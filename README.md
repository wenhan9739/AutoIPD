# AtuoIPDR (formerly IPDR-KM2IPD): End-to-End Reconstruction of Individual Patient Data from Published Oncology Trials

**Version 1.1**

An open-source pipeline that takes published randomized controlled trial (RCT) documents (PDF, docx, pptx) as input and produces quality-controlled, reconstructed individual patient data (IPD) from published Kaplan–Meier (KM) survival curves — with per-arm population and intervention annotations.

> Use case: meta-analyses and IPD analyses where only published KM curves are available, enabling automated reconstruction of pseudo-IPD for survival analysis.

## Pipeline Overview (5 Stages)

```
Published RCT documents (PDF / docx / pptx)
   │  Stage 1: MinerU batch parsing (layout + chart detection + text/figure extraction)
   ▼
Stage 2: Candidate-page selection (caption/keyword/vector-density matching) → manifest.json
   ▼
Stage 3: KM curve digitization engine
   │    Panel detection (L-shaped axis pairing + container removal)
   │    Axis calibration (vector text-layer ticks / geometric tick-marks + OCR, RANSAC fitting)
   │    Curve extraction (vector stroke paths / raster hue clustering + column tracking)
   │    KM prior enforcement (start at (0,1), monotone non-increasing, censor-mark repair)
   ▼
Stage 4: IPD reconstruction — Guyot (2012) algorithm, dual implementation
   │    Python port (kmfig/guyot.py, line-faithful to IPDfromKM)
   │    R reference (IPDfromKM::preprocess + getIPD, same inputs)
   ▼
Stage 5: Quality control (4 layers)
        Data integrity / trace precision / three-way curve comparison / published-value closure
        + population/intervention master table + per-panel visual QA images
```

## Validation Results (27 RCTs, 477 curves)

| Metric | Value |
|---|---|
| Trials covered | 27 / 27 |
| Digitized curves | 477 (233 vector + 244 raster) |
| Trace precision (median) | 1.000 (427/474 arms ≥ 0.9) |
| Reconstructed panels | 78 |
| Reconstructed arms | 180 (91 OS + 64 PFS + 25 other) |
| Py vs R IPD row-identical | 141/180 (78.3%) |
| Reconstructed KM vs original RMSE | median 0.0092, max 0.0495 |
| Reconstructed median vs digitized median | median absolute difference 0.000 months |
| Cox HR concordance (2 trials) | KEYNOTE-189 PFS: 2.04 [1.71–2.43] ∈ published CI; AK105-302: 2.34 ≈ 2.33 |

## Pipeline Stages in Detail

### Stage 1–2: Document parsing and candidate-page selection

MinerU (v3.2, pipeline backend, GPU) parses all documents into per-page layout trees with text, chart/image detection, and cropped figures. Candidate pages are shortlisted by three signals: chart-type blocks from MinerU's chart detector, caption or page text matching a curated keyword set (Kaplan–Meier, OS, PFS, EFS, DFS, DOR), and pages without a text layer but with dense vector graphics, which catches born-digital figures that layout models miss.

### Stage 3: Curve digitization engine

**Panel detection** (`kmfig/panels.py`): Dark low-saturation pixel masks (excluding colored curves and light gridlines) are processed with morphological opening to find long horizontal/vertical line segments. L-shaped axis systems are paired by junction matching. Container frames (enclosing smaller panels) are discarded.

**Axis calibration** (`kmfig/calibrate.py`, dual route):
- Vector PDFs: tick-label positions read directly from the text layer (sub-pixel accuracy)
- Raster/scanned: geometric tick-mark detection + adaptive-window OCR (RapidOCR), with projection-segmentation fallback
- Robust fitting: RANSAC initialization + iterative outlier rejection (OCR misreads don't corrupt calibration)
- Equidistance checking with monotone-consistency fallback (handles fraction axes with partial label coverage)
- Both percentage (0–100) and fraction (0–1) axes auto-detected

**Curve extraction** (three tiers):
1. Vector mode: color-grouped stroke paths → endpoint-hash chaining (successor-preferring at junctions to avoid censor-mark truncation) → fragment merging with endpoint-distance + KM-monotonicity constraints → spur/terminal-tick trimming → border-frame exclusion
2. Raster mode: HSV saturated pixels (s_min=0.45 to exclude pale CI bands, fallback to 0.30) → hue local-maxima clustering → connected-component union (text-like blobs removed) → column-wise tracking (nearest-to-previous, strict no-upward rule) → censor-mark back-fill → truncation at last genuine pixel → monotonicity enforcement
3. Special fallbacks: near-black monochrome arms (character-like blob filter + tracking), pale dotted arms (reduced saturation + per-dot component threading)

**KM priors enforced throughout**: curves must start at (0, 1), be monotonically non-increasing, terminate at the last genuine pixel; duplicate arms (traces coinciding within 3× line width for >90% of columns) are deduplicated.

### Stage 4: IPD reconstruction (Guyot algorithm, dual implementation)

The Python implementation (`kmfig/guyot.py`) is a line-faithful port of IPDfromKM 0.1.10, reproducing its initialization, R-style division semantics, half-even rounding of event counts, and uniform within-interval placement of censored observations. The R implementation calls IPDfromKM's `preprocess()` and `getIPD()` directly on the same inputs. Both are cross-validated: 141/180 arms produce row-identical IPD tables; the remaining 39 differ only in individual censor-time placement (KM-curve RMSE ≤ 0.007 between implementations).

### Stage 5: Quality control

Four automated layers, with per-panel visual-QA images for human review:

1. **Data integrity**: output CSVs are compared row-by-row with the reconstruction JSON; times must be monotone, survival in [0,1], and the first vertex must equal (0, 1)
2. **Trace precision**: fraction of sampled trace points within 2.5 px of a genuine curve pixel
3. **Three-way comparison**: reconstructed KM curves from both implementations are overlaid on the original digitized curves, with per-panel RMSE, median differences, and milestone survival agreement
4. **Closure against the source publication**: medians and HRs printed in the figure or reported in the text are matched to each arm and compared with the digitized/reconstructed values

Per-arm population and intervention annotations are assigned through: (a) same-color text matching (NEJM/JCO style), (b) figure annotation matched by median proximity (Lancet style), (c) legend swatch + black-text OCR (raster figures), backed by a curated 27-trial treatment map.

## Usage

```bash
pip install -r requirements.txt
# R side: install.packages(c("IPDfromKM", "survival", "jsonlite"))

export IPDR_ROOT=/path/to/workdir
mkdir -p $IPDR_ROOT/trials    # organize PDFs by trial name

cd pipeline
python run_mineru_all.py       # Stage 1: batch MinerU (resumable)
python select_figures.py       # Stage 2: candidate-page manifest
python digitize_all.py         # Stage 3: full digitization + QA
python annotate_panels.py      # Stage 3b: population/intervention annotation + OCR risk tables
python ipd_reconstruct.py      # Stage 4: Python Guyot batch reconstruction
Rscript ipdfromkm_batch.R \    # Stage 4: R IPDfromKM (same inputs)
    "$IPDR_ROOT/results/ipd/r_input" "$IPDR_ROOT/results/ipd/r_out"
python compare_ipd.py          # Stage 5: three-way comparison
python verify_all.py           # Stage 5: quantitative verification
python qa_ipd_panels.py        # Stage 5: per-panel visual QA images
python build_master.py         # Stage 5: population/intervention master table
```

Set `IPDR_ROOT` to your working directory (contains trial PDFs, mineru_out/, work/, results/).

## Output Structure

```
$IPDR_ROOT/
├── mineru_out/<trial>/<doc>/     # MinerU parsing results
├── work/verify/                  # Per-arm trace verification images
├── results/
│   ├── curves_master.csv         # ★ Master table: trial/population/intervention/role/median/QA
│   ├── QA_report.md              # QC report (flagged items for human review)
│   ├── verification.csv          # Per-arm quantitative verification
│   ├── overlays/                 # Digitized vs published overlay images
│   └── ipd/
│       ├── <panel>__<arm>.ipd.csv  # ★ Reconstructed pseudo-IPD (time, event)
│       ├── plots/                  # Reconstructed KM vs original overlay
│       ├── key_values.csv          # Milestone survival / medians / risk-table errors
│       ├── compare_report.csv      # Py/R/original three-way comparison
│       └── r_out/                  # R IPDfromKM results + Cox HR
```

## Key Output: curves_master.csv

One row per arm with: trial, source document, page, panel, endpoint (OS/PFS/DOR), population, intervention (standardized regimen), role (experimental/control), reconstruction mode, median survival, published-median match, milestone survival rates (S6–S60), trace precision, risk-table error, QA flags, and file paths to the IPD CSV and QA image.

## Known Limitations

- Vector PDFs use exact path extraction (precision ≈ 0.05 months); raster/scanned figures use 300-dpi rendering (precision limited by source resolution)
- Long overlapping stretches between arms introduce ~half a line-width of bias (< 1% survival); flagged by the overlap metric
- KM figures printed as two solid black lines without color or line-style differences cannot be separated automatically
- Tail plateaus with few remaining patients can shift medians by > 0.5 months due to Guyot integer rounding (7/172 arms observed)
- Heavily overlapping multi-arm supplementary figures may have tracking artifacts; flagged by QA and listed for human review

## AI-Assisted Development Disclosure

The pipeline (~6,000 lines of Python and R across 25 scripts) was developed through human-AI collaboration using the GLM-5.3-Flash model within the ZCode harness. The development session spanned approximately 24 calendar hours, comprising 28 user turns and 1,310 model requests. Total token consumption was 446,549,976 (445.4M input, of which 97.5% served from cache; 1.15M output), with 12.7 hours of cumulative model processing time. At official list prices (US$0.15 per million fresh-input tokens, US$0.026 per million cache-read tokens, US$0.50 per million output tokens), the total development cost was approximately US$13.52. Four parallel visual-verification sub-agents consumed an additional 6.3M tokens (≈ US$0.36).

## References

- Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. BMC Med Res Methodol. 2012;12:9.
- IPDfromKM R package (CRAN: IPDfromKM) — algorithm reference implementation.
- MinerU: https://github.com/opendatalab/MinerU
- RapidOCR: https://github.com/RapidAI/RapidOCR

## License

GPL-3.0 (see LICENSE file)
