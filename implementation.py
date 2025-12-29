"""
Horizontal Visibility–based Independent Spatial Row Reconstruction Algorithm
This script implements the "Horizontal Visibility–based Independent Spatial Row Reconstruction" algorithm for parsing complex PDF tables. It analyzes table geometry to detect logical columns via x-center clustering, parses multi-level headers using x-range containment, flattens non-header elements, and reconstructs rows using anchor-based y-scope division and horizontal visibility lines.
Mathematical Concepts:

1D Clustering for Cardinality: X-centers cx = (x0 + x1)/2 clustered with distance |cx_i - cx_j| ≤ x_tol; centroids as mean of cluster points.
Interval Containment for Headers: Child [c_x0, c_x1] ⊆ parent [p_x0, p_x1] if p_x0 ≤ c_x0 and c_x1 ≤ p_x1.
Y-Height Ratio for Row Count: num_rows = floor(anchor_h / min_h), where anchor_h = a_y1 - a_y0, min_h = min over right columns of min element heights.
Interval Overlap for Visibility: Element [e_y0, e_y1] overlaps band [s_y0, s_y1] if e_y0 < s_y1 and s_y0 < e_y1.
Containment for Scope: Element [e_y0, e_y1] within anchor [a_y0, a_y1] if a_y0 ≤ e_y0 and e_y1 ≤ a_y1.
Overlap for Column Assignment: Max ov = max(0, min(el_x1, h_x1) - max(el_x0, h_x0)); assign if ov / el_w > 0.5, where el_w = el_x1 - el_x0.

No line equations used; all operations on 1D projections of 2D AABBs (axis-aligned bounding boxes). Geometry: Projections reduce 2D to 1D intervals for containment/overlap checks.
Author :  JUSTIN.M
"""
import pymupdf
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


# =========================================================
# 1. Cardinality detection (unchanged)
# =========================================================

def compute_column_slots(table, x_tol=3.0):
    """
    Clusters x-centers of table cells to determine logical column slots.
    Logical Explanation: Collects midpoints of all cell bounding boxes, sorts them, and groups into clusters based on proximity within x_tol. Computes average for each cluster as slot position.
    Mathematical Explanation: For each cell Rect r = [x0, y0, x1, y1], compute cx = (x0 + x1)/2. Sort cx list. Cluster: Start with slot_0 = [cx_0]; for cx_i, if |cx_i - last in current slot| ≤ x_tol, append; else new slot. Centroid c = Σ x / n for cluster of n points. This is threshold-based agglomerative clustering in 1D, using absolute distance metric d(a,b) = |a - b|.
    """
    xs = []
    for row in table.rows:
        for cell in row.cells:
            if cell is None:
                continue
            r = pymupdf.Rect(cell)
            xs.append((r.x0 + r.x1) / 2)
    if not xs:
        return []
    xs.sort()
    slots = [[xs[0]]]
    for x in xs[1:]:
        if abs(x - slots[-1][-1]) <= x_tol:
            slots[-1].append(x)
        else:
            slots.append([x])
    return [sum(c) / len(c) for c in slots]


def count_row_columns(row, col_centers, x_tol=3.0):
    """
    Counts occupied logical columns in a single row by projecting cell centers to slots.
    Logical Explanation: For each cell in the row, calculate its x-center and assign to the nearest slot if within tolerance, using a set to count unique occupations.
    Mathematical Explanation: Cell center cx = (r.x0 + r.x1)/2. For each slot col_x, check if |cx - col_x| ≤ x_tol (tolerance-based nearest neighbor). Occupied count = |set of assigned i|. This is 1D projection and binning with tolerance.
    """
    occupied = set()
    for cell in row.cells:
        if cell is None:
            continue
        r = pymupdf.Rect(cell)
        cx = (r.x0 + r.x1) / 2
        for i, col_x in enumerate(col_centers):
            if abs(cx - col_x) <= x_tol:
                occupied.add(i)
                break
    return len(occupied)


def detect_cardinality_most_frequent(table):
    """
    Determines the logical number of columns (cardinality) as the most frequent row occupancy.
    Logical Explanation: Computes column slots, then for each row counts occupied slots, and selects the mode frequency as cardinality.
    Mathematical Explanation: Frequency count using Counter on n = count_row_columns per row. Mode = argmax freq[n]. Empirical mode of discrete distribution over occupancy counts.
    """
    col_centers = compute_column_slots(table)
    freq = Counter()
    for row in table.rows:
        n = count_row_columns(row, col_centers)
        if n > 0:
            freq[n] += 1
    print("Row column-count frequency:", freq)
    return freq.most_common(1)[0][0]


# =========================================================
# 2. Header detection (deterministic, unchanged but extended to return ranges)
# =========================================================

def build_header_cells(table, text_rows, row_idx):
    """
    Determines the logical number of columns (cardinality) as the most frequent row occupancy.
    Logical Explanation: Computes column slots, then for each row counts occupied slots, and selects the mode frequency as cardinality.
    Mathematical Explanation: Frequency count using Counter on n = count_row_columns per row. Mode = argmax freq[n]. Empirical mode of discrete distribution over occupancy counts.
    """
    cells = []
    if row_idx >= len(table.rows):
        return cells
    for c_idx, cell in enumerate(table.rows[row_idx].cells):
        if cell is None:
            continue
        r = pymupdf.Rect(cell)
        txt = text_rows[row_idx][c_idx] if c_idx < len(text_rows[row_idx]) else ""
        cells.append({
            "text": " ".join(txt.split()),
            "x0": r.x0,
            "x1": r.x1,
            "y0": r.y0,
            "y1": r.y1,
        })
    return sorted(cells, key=lambda x: x["x0"])


def parse_headers_deterministic(row0, row1, cardinality):
    """
    Deterministically parses and merges headers from first two rows based on x-containment.
    Logical Explanation: If row0 matches cardinality, use it. Else, merge children into parents if contained, append unmatched, pad if short.
    Mathematical Explanation: Containment: for child c, parent p, check p.x0 ≤ c.x0 and c.x1 ≤ p.x1 (1D interval inclusion). Sort children by x0 for order. Padding duplicates last range [x0, x1].
    """
    if len(row0) == cardinality:
        return [{"text": c["text"], "x0": c["x0"], "x1": c["x1"]} for c in row0]

    headers = []
    used = set()

    for p in row0:
        children = []
        for i, c in enumerate(row1):
            if i in used:
                continue
            # Containment: child's x-range within parent's
            if p["x0"] <= c["x0"] and c["x1"] <= p["x1"]:
                children.append(c)
                used.add(i)
        if children:
            # Sort children by x0 for sequential assignment
            children.sort(key=lambda c: c["x0"])
            for c in children:
                headers.append({
                    "text": f"{p['text']} / {c['text']}",
                    "x0": c["x0"],
                    "x1": c["x1"]
                })
        else:
            headers.append({
                "text": p["text"],
                "x0": p["x0"],
                "x1": p["x1"]
            })

    for i, c in enumerate(row1):
        if i not in used:
            headers.append({
                "text": c["text"],
                "x0": c["x0"],
                "x1": c["x1"]
            })

    # Pad if necessary (duplicate last range for unnamed columns)
    while len(headers) < cardinality:
        last = headers[-1]
        headers.append({
            "text": f"column_{len(headers)+1}",
            "x0": last["x0"],
            "x1": last["x1"]
        })

    return headers[:cardinality]


# =========================================================
# 3. Flatten all non-header elements
# =========================================================

def flatten_elements(table, text_rows, header_y_max):
    """
    Flattens non-header table content into list of elements with text and bounding boxes.
    Logical Explanation: Skip header band (y1 <= header_y_max), collect text-normalized elements with height h and row index.
    Mathematical Explanation: Height h = y1 - y0 (interval length). Filter y1 > header_y_max (vertical threshold). r_idx for tracking visual rows.
    """
    elements = []
    for r_idx, row in enumerate(table.rows):
        texts = text_rows[r_idx]
        for c_idx, cell in enumerate(row.cells):
            if cell is None:
                continue
            rect = pymupdf.Rect(cell)
            if rect.y1 <= header_y_max:
                continue
            text = texts[c_idx] if c_idx < len(texts) else ""
            text = " ".join(text.split())
            if not text:
                continue
            elements.append({
                "text": text,
                "x0": rect.x0,
                "x1": rect.x1,
                "y0": rect.y0,
                "y1": rect.y1,
                "h": rect.y1 - rect.y0,
                "r_idx": r_idx,
            })
    return elements


# =========================================================
# 4. Row reconstruction (corrected seed logic)
# =========================================================

def assign_header(el, header_ranges):
    """
    Assigns an element to a header column based on maximum x-overlap.
    Logical Explanation: Compute overlap with each header range, select max if >50% of element width.
    Mathematical Explanation: Overlap ov = max(0, min(el_x1, x1) - max(el_x0, x0)). Ratio ov / (el_x1 - el_x0) > 0.5. This is 1D interval overlap length normalized by element width for affinity.
    """
    best_i = None
    max_overlap = 0.0
    el_x0, el_x1 = el["x0"], el["x1"]
    el_w = el_x1 - el_x0
    for i, (x0, x1) in enumerate(header_ranges):
        o_l = max(el_x0, x0)
        o_r = min(el_x1, x1)
        ov = max(0.0, o_r - o_l)
        if ov > max_overlap:
            max_overlap = ov
            best_i = i
    if best_i is not None and max_overlap / el_w > 0.5:
        return best_i
    return None


def overlaps_y(e_y0, e_y1, s_y0, s_y1):
    """
    Checks if two y-intervals overlap.
    Logical Explanation: Simple test for any intersection between element and seed band.
    Mathematical Explanation: Intervals [e_y0, e_y1], [s_y0, s_y1] overlap if e_y0 < s_y1 and s_y0 < e_y1 (standard 1D overlap condition, equivalent to not (e_y1 ≤ s_y0 or s_y1 ≤ e_y0)).
    """
    return e_y0 < s_y1 and s_y0 < e_y1


def y_fully_within(e_y0, e_y1, a_y0, a_y1):
    """
    Checks if element y-interval is fully contained within anchor scope.
    Logical Explanation: Strict containment for scoping elements to anchor.
    Mathematical Explanation: [e_y0, e_y1] ⊆ [a_y0, a_y1] if a_y0 ≤ e_y0 and e_y1 ≤ a_y1 (1D interval inclusion).
    """
    return a_y0 <= e_y0 and e_y1 <= a_y1


def reconstruct_rows(elements, header_dicts, header_ranges, header_y_max, table, text_rows):
    """
    Reconstructs logical rows independently per anchor using seed-based horizontal visibility.
    Logical Explanation: Select anchors, infer spans via empty cells, define y-scope, compute row count via height ratio, use seeds for bands, collect overlapping elements, assign to columns, normalize output.
    Mathematical Explanation: Anchor h = y1 - y0. Row count = max(1, floor(anchor_h / min_h)). Bands from seeds or uniform step = anchor_h / row_count. Elements via y-overlap. Assignment via x-overlap. Span injection if overlap(anchor_scope, band) but not overlap(anchor_bbox, band).
    """
    rows = []
    anchor_idx = 0  # left-most column

    potential_anchors = [
        e for e in elements
        if assign_header(e, header_ranges) == anchor_idx and e["text"].strip() != ""
    ]
    potential_anchors.sort(key=lambda e: e["r_idx"])

    header_names = [h["text"] for h in header_dicts]
    num_table_rows = len(table.rows)

    processed_up_to = -1

    for anch in potential_anchors:
        start_r = anch["r_idx"]
        if start_r <= processed_up_to:
            continue

        end_r = start_r
        while end_r + 1 < num_table_rows:
            next_row = table.rows[end_r + 1]
            next_texts = text_rows[end_r + 1]
            next_cell = next_row.cells[anchor_idx] if anchor_idx < len(next_row.cells) else None
            next_text = next_texts[anchor_idx] if anchor_idx < len(next_texts) else ""
            if next_cell is None or next_text.strip() == "":
                end_r += 1
            else:
                break

        effective_a_y0 = max(anch["y0"], header_y_max)
        # Fix: get y1 from a non-None cell in end_r
        end_row = table.rows[end_r]
        effective_a_y1 = None
        for cell in end_row.cells:
            if cell is not None:
                effective_a_y1 = pymupdf.Rect(cell).y1
                break
        if effective_a_y1 is None:
            effective_a_y1 = anch["y1"]  # fallback
        anchor_h = effective_a_y1 - effective_a_y0

        # Closed scope: elements fully within effective anchor scope
        scope = [
            e for e in elements
            if y_fully_within(e["y0"], e["y1"], effective_a_y0, effective_a_y1)
        ]
        if not scope:
            processed_up_to = end_r
            continue

        # Find overall min_h in right columns
        col_min_hs = defaultdict(lambda: float('inf'))
        for e in scope:
            h_idx = assign_header(e, header_ranges)
            if h_idx is None or h_idx <= anchor_idx:
                continue
            col_min_hs[h_idx] = min(col_min_hs[h_idx], e["h"])

        print(f"\nAnchor '{anch['text']}' (rows {start_r}-{end_r})")

        if col_min_hs:
            overall_min_h = min(col_min_hs.values())
            candidate_cols = [col for col, mh in col_min_hs.items() if abs(mh - overall_min_h) < 1e-6]
            seed_column = min(candidate_cols)  # nearest

            # Seeds in chosen column with approx min_h
            seeds = [
                e for e in scope
                if assign_header(e, header_ranges) == seed_column
                and abs(e["h"] - overall_min_h) < 1.0
            ]
            seeds.sort(key=lambda e: e["y0"])

            row_count = max(1, int(anchor_h / overall_min_h))

            print(f" → seed_col={seed_column}, min_h={overall_min_h}, rows={row_count}, physical_seeds={len(seeds)}")

            step = anchor_h / row_count
            for i in range(row_count):
                if i < len(seeds):
                    s_y0, s_y1 = seeds[i]["y0"], seeds[i]["y1"]
                else:
                    s_y0 = effective_a_y0 + i * step
                    s_y1 = effective_a_y0 + (i + 1) * step

                row_elems = [
                    e for e in elements
                    if overlaps_y(e["y0"], e["y1"], s_y0, s_y1)
                ]

                # Special handling for anchor span (if bbox not full)
                span_overlap = overlaps_y(effective_a_y0, effective_a_y1, s_y0, s_y1)
                anch_overlap = overlaps_y(anch["y0"], anch["y1"], s_y0, s_y1)
                if span_overlap and not anch_overlap:
                    row_elems.append(anch)

                buckets = defaultdict(list)
                for e in row_elems:
                    h_idx = assign_header(e, header_ranges)
                    if h_idx is not None:
                        buckets[h_idx].append((e["text"], e["x0"]))

                parts = []
                for j, h in enumerate(header_names):
                    if j in buckets:
                        col_items = sorted(buckets[j], key=lambda t: t[1])  # sort by x0
                        for text, _ in col_items:
                            parts.append(f"{h}:{text}")

                if parts:
                    row_str = ",".join(parts)
                    rows.append(row_str)
                    print("-" * 100)
                    print(row_str)
        else:
            # Fallback for rows with no right elements (e.g., only left content or checkboxes without text)
            print(" → fallback single row")
            row_count = 1
            s_y0, s_y1 = effective_a_y0, effective_a_y1
            row_elems = [
                e for e in elements
                if overlaps_y(e["y0"], e["y1"], s_y0, s_y1)
            ]
            buckets = defaultdict(list)
            for e in row_elems:
                h_idx = assign_header(e, header_ranges)
                if h_idx is not None:
                    buckets[h_idx].append((e["text"], e["x0"]))
            parts = []
            for j, h in enumerate(header_names):
                if j in buckets:
                    col_items = sorted(buckets[j], key=lambda t: t[1])
                    for text, _ in col_items:
                        parts.append(f"{h}:{text}")
            if parts:
                row_str = ",".join(parts)
                rows.append(row_str)
                print("-" * 100)
                print(row_str)

        processed_up_to = end_r

    return rows


# =========================================================
# 5. Driver
# =========================================================

def process_pdf_tables(pdf_path):
    """
    Driver function to process all tables in a PDF, applying the full parsing pipeline.
    Logical Explanation: Open PDF, find tables per page, detect cardinality, extract text, parse headers, flatten elements, reconstruct and print rows.
    Mathematical Explanation: Aggregates per-table results; no new math, orchestrates prior functions.
    """
    doc = pymupdf.open(pdf_path)

    for page_idx, page in enumerate(doc):
        tf = page.find_tables()
        for t_idx, table in enumerate(tf.tables):
            print("\n" + "=" * 120)
            print(f"Page {page_idx}, Table {t_idx}")

            cardinality = detect_cardinality_most_frequent(table)
            text_rows = table.extract()

            row0 = build_header_cells(table, text_rows, 0)
            row1 = build_header_cells(table, text_rows, 1)

            header_dicts = parse_headers_deterministic(row0, row1, cardinality)
            header_ranges = [(h["x0"], h["x1"]) for h in header_dicts]

            header_y_max = max([c["y1"] for c in row0 + row1] + [0])

            print("\nGenerated Headers:")
            for h in header_dicts:
                print(" -", h["text"])

            elements = flatten_elements(table, text_rows, header_y_max)
            rows = reconstruct_rows(elements, header_dicts, header_ranges, header_y_max, table, text_rows)

            print("\nReconstructed rows:")
            for r in rows:
                print(r)


if __name__ == "__main__":
    process_pdf_tables("aplio_trim.pdf")
