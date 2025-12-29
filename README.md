# Horizontal Visibility–based Independent Spatial Row Reconstruction

A lightweight, purely geometric algorithm for reconstructing logical rows from complex tables in PDFs or document images.

This repository implements a novel approach to **table structure recognition (TSR)** that avoids deep learning, grid assumptions, and fragile heuristics. It works directly on a list of atomic text elements with bounding boxes — the exact output format produced by most OCR and document layout models (LayoutLM, Donut, TableFormer, PubLayNet detectors, etc.).

## What It Solves

Extracting structured data from **real-world complex tables** is notoriously difficult:

- Multi-level (nested) headers
- Row spans that cross many visual rows
- Column spans
- Multi-line cells
- Irregular row heights
- Footnotes and remarks that leak across rows
- Noisy or imperfect bounding boxes from OCR/DL detectors

Traditional grid-centric methods (e.g., Camelot, Tabula, most early OCR pipelines) break when cells merge or rows are uneven. Deep learning models (e.g., TableTransformer) are heavy, opaque, and often overkill for post-processing.

**This algorithm** flips the problem:  
It **starts from the leftmost column** (anchors), infers independent vertical scopes, and uses **horizontal visibility bands** (y-interval overlaps) to group elements into logical rows — all with pure 1D interval algebra on x- and y-projections.

## Why It Is Good

- **Lightweight & fast**: O(N log N) time (N = number of text elements), no GPU, no training
- **Explainable**: Every decision is traceable via simple geometric rules (no black-box DL)
- **Robust to spans**: Rowspans are handled natively via anchor scope extension — no fake rows or cascading errors
- **Tolerant to noise**: Floating-point tolerances (1 pt) and overlap thresholds (>50%) handle PDF rendering jitter and DL bbox errors
- **Generic input**: Accepts any list of `[{"text": str, "bbox": (x0, y0, x1, y1)}]` — integrates easily as a post-processor for LayoutLM, Donut, OCR engines, etc.
- **RAG-ready output**: Produces normalized sentences ("Header1:value1,Header2:value2") perfect for embedding and retrieval
- **No external dependencies**: Pure Python + `collections` (optional `numpy` for speed)

## How the Algorithm Works (Core Math Explained)

The algorithm reduces 2D table layout to **1D interval algebra** on x- and y-projections of axis-aligned bounding boxes.

### 1. Infer Logical Cardinality (Number of Columns)
- Compute x-centers: cx = (x₀ + x₁)/2 for every element
- Sort centers, cluster with distance threshold (median gap × 1.5)
- For each approximate row (y-center clusters), project cx to nearest cluster centroid
- Cardinality = most frequent number of occupied columns across rows  
  → Mode of occupancy counts (robust to partial rows)

### 2. Infer and Merge Headers (Deterministic)
- Sort elements by y-center → cluster into top ~2 visual rows (gap-based)
- Merge children into parents via strict x-containment:  
  child [c_x₀, c_x₁] ⊆ parent [p_x₀, p_x₁] ⇔ p_x₀ ≤ c_x₀ ∧ c_x₁ ≤ p_x₁
- Record final header ranges [x₀, x₁] for column assignment
- Header band: max y₁ of header elements

### 3. Flatten Non-Header Elements
- Keep only elements with y₁ > header_y_max
- Compute height h = y₁ - y₀ for each

### 4. Independent Anchor-based Row Reconstruction (Core Innovation)
1. **Anchors**: Elements assigned to column 0 (via max x-overlap ratio > 0.5)
   - Overlap ov = max(0, min(e_x₁, h_x₁) - max(e_x₀, h_x₀))
   - Assign if ov / (e_x₁ - e_x₀) > 0.5

2. **Scope per anchor**: Vertical band [effective_a_y₀, a_y₁] where effective_a_y₀ = max(anchor y₀, header_y_max)
   - Scope elements: those fully contained (a_y₀ ≤ e_y₀ ∧ e_y₁ ≤ a_y₁)

3. **Row cardinality**:  
   - min_h = min over right columns (col > 0) of min element height in scope  
   - num_rows = max(1, ⌊(a_y₁ - effective_a_y₀) / min_h⌋)

4. **Seeds**: In the column with smallest min_h (nearest if ties), elements with |h - min_h| < 1 pt, sorted by y₀

5. **Visibility bands**: For each logical row i:
   - Use seed[i] band [s_y₀, s_y₁] if available
   - Else uniform step = anchor height / num_rows
   - Collect elements overlapping band: e_y₀ < s_y₁ ∧ s_y₀ < e_y₁

6. **Anchor injection**: If band overlaps scope but not anchor bbox, add anchor (handles tall spanning cells)

7. **Column assignment & normalization**: Max overlap > 50% → "header:text" per column, sorted by x₀ within column

### Fallbacks
- No right-column elements → single full-band row
- Fewer seeds than rows → uniform division

All geometry is **1D interval algebra** on projections — no line equations, no 2D areas, no topology.

## Installation & Usage

```bash
pip install -r requirements.txt  # only collections, typing (optional numpy)
