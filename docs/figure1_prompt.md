# Figure 1 Prompt — Method Overview (Left-Right Layout)

A clean academic method figure (MICCAI style). Flat vector graphics, white background.
ALL arrows strictly horizontal or vertical. No curves, no diagonals. No decorative elements.
NO rail labels, NO numbered headers, NO routing annotations.

Layout: TWO side-by-side panels separated by a thin vertical gray line.
- Panel (a) on LEFT (~55% width): Main pipeline, vertical top-to-bottom flow.
- Panel (b) on RIGHT (~45% width): L_z inter-slice diversity loss, vertical flow.
- Panel labels "(a)" and "(b)" at top-left corner of each panel, bold sans-serif.

Overall size: ~17 cm wide × ~10 cm tall, suitable for academic full-width figure.

=========================================
Panel (a): Noise-Slice Protection Pipeline
=========================================

Vertical top-to-bottom flow. Each processing step is a row. Arrows go downward between rows.

--- ROW 1: Input ---
Two items side by side, inside a light gray (#F5F5F5) rounded rectangle:
  LEFT: "x" — a small stack of 3–4 overlapping axial CT slices (grayscale), thin black border.
  RIGHT: "y" — a matching small stack of label masks (black background, white organ contours).
  Small italic label below the region: "input volume & label"

--- ↓ arrow from x downward ---

--- ROW 2: Noise Generator ---
[G_φ] — orange-red (#E8553A) rounded rectangle, flat 🔥 icon on left, white bold label "G_φ" on right.
  Small gray annotation to the right: "tanh · ε"

--- ↓ arrow downward, labeled "δ_raw" ---

--- ROW 3: ROI Masking ---
[⊙] — a small circle with "⊙" symbol (element-wise multiply).
  A SHORT horizontal arrow comes from the RIGHT, from y, into ⊙.
  This arrow is labeled "M_roi = I(y>0)" in small gray text.
  (y is at the same vertical level in the input region — draw a path:
   from y going DOWN to this row level, then LEFT into ⊙.
   Keep this path close to the main flow, not far away.)

--- ↓ arrow downward ---

--- ROW 4: Clamping ---
[Clamp [−ε, ε]] — small white rounded rectangle with thin border.

--- ↓ arrow downward, labeled "δ" in yellow-green (#A8D86E) pill ---

--- ROW 5: Perturbation Addition ---
[⊕] — a small circle with "+" symbol.
  A SHORT horizontal arrow comes from the LEFT, from x, into ⊕.
  (Draw a path from x: going DOWN along the left side of the panel to this row, then RIGHT into ⊕.
   This is a single vertical line on the left margin with one rightward branch. Keep it thin and gray.)

--- ↓ arrow downward ---

--- ROW 6: Perturbed Output ---
Inside a light green (#E8F5E9) rounded rectangle:
  "x^u" — CT slice stack with green border, flat 🔒 icon.
  Small label below: "x + δ"

--- ↓ arrow downward ---

--- ROW 7: Surrogate Model + Losses ---
Inside a light blue (#E3F2FD) rounded rectangle:

  [F_θ] — steel-blue (#4682B4) rounded rectangle, flat ❄️ icon, white label "F_θ".
    A horizontal arrow comes from the LEFT into F_θ, labeled "x (no grad)" in small gray italic.
    (This arrow branches from the same left-margin vertical line as the x→⊕ path above,
     continuing down and branching RIGHT into F_θ. Use dashed gray style for this arrow.)

  Below F_θ, three loss boxes arranged horizontally:
    [L_seg] — small white box, annotation "DiceCE" below
    [L_spec] — small white box, annotation "FFT-L1" below
    [L_z] — small white box, annotation "→(b)" below, with a dashed arrow pointing RIGHT toward Panel (b)

  A horizontal arrow from y goes into L_seg (for ground truth supervision).
  (This arrow branches from the y's downward path on the right margin, going LEFT into L_seg.)

  Below losses, the formula:
  L = L_seg + λ_s · L_spec + λ_z · L_z

--- AUXILIARY PATHS SUMMARY (Panel a) ---

There are exactly TWO vertical margin paths:

1. LEFT MARGIN (from x): A thin gray vertical line running down the left side of the panel.
   It branches RIGHT at two points:
   - At ROW 5 → into ⊕ (solid arrow)
   - At ROW 7 → into F_θ (dashed arrow, labeled "no grad")

2. RIGHT MARGIN (from y): A thin gray vertical line running down the right side of the panel.
   It branches LEFT at two points:
   - At ROW 3 → into ⊙ (solid arrow, labeled "M_roi")
   - At ROW 7 → into L_seg (solid arrow)

These margin paths are thin, gray, and unobtrusive. NO labels on the vertical segments themselves.

=========================================
Panel (b): L_z — Inter-Slice Spectral Diversity
=========================================

Vertical top-to-bottom flow inside a light orange (#FFF3E0) rounded rectangle.
Bold title at top: "L_z : Inter-Slice Diversity"

--- STEP 1: Input noise ---
[δ] — yellow-green (#A8D86E) pill shape, bold "δ".
A dashed arrow comes from Panel (a) Row 4 (the δ output) pointing RIGHT into this panel.

--- ↓ arrow, label "slice along z" ---

--- STEP 2: Individual 2D slices ---
4 small squares in a horizontal row, each showing a DIFFERENT 2D noise pattern (grayscale, random-looking).
Labels below: "δ^(z)", "δ^(z+1)", "δ^(z+2)", "δ^(z+3)"

--- ↓ arrow, label "2D FFT per slice" ---

--- STEP 3: Frequency magnitude spectra ---
4 small squares in a horizontal row, each showing a DIFFERENT colorful 2D frequency spectrum
(bright center, radial falloff pattern, each visually distinct from neighbors).
Labels below: "S_z", "S_{z+1}", "S_{z+2}", "S_{z+3}"

Between each adjacent pair of spectra, a bold red "≠" symbol to emphasize they are different.

--- ↓ arrow ---

--- STEP 4: Pairwise distance ---
White rounded rectangle:
"‖S_{z+1} − S_z‖₂"

--- ↓ arrow ---

--- STEP 5: Aggregation ---
White rounded rectangle:
"Mean over z"

--- ↓ arrow ---

--- STEP 6: Final loss ---
[L_z] — white box with orange border, bold text.
Formula below:
L_z = − 1/(D−1) · Σ ‖S_{z+1} − S_z‖₂

--- BOTTOM ANNOTATION ---
Below the main flow, a small annotation box with gray background:
  Left: gray cube icon labeled "3D Conv"
  Right: italic text "contradictory spectral patterns across slices → model fails to learn inter-slice consistency"
  Red ✗ icon at the end.

=========================================
Style Guide
=========================================

Colors:
  Orange-red #E8553A — G_φ (noise generator)
  Steel-blue #4682B4 — F_θ (surrogate model)
  Yellow-green #A8D86E — δ (perturbation)
  Light gray #F5F5F5 — input background
  Light green #E8F5E9 — output background
  Light blue #E3F2FD — loss region background
  Light orange #FFF3E0 — Panel (b) background
  Gray #999 — auxiliary paths and annotations

Typography:
  Sans-serif font (Helvetica/Arial style).
  Math variables in italic.
  Bold for block labels (G_φ, F_θ, L_z, etc.).
  Small gray text for annotations.

Blocks:
  Processing blocks: rounded rectangles with colored fill and white text.
  Operations (⊙, ⊕, Clamp): small white circles/boxes on the flow path.
  Loss boxes: white with thin border.

Icons (🔥❄️🔒): flat vector style, NOT emoji. Simple, monochrome-friendly.

Arrows:
  Main flow: black, medium weight, downward.
  Auxiliary paths: thin gray.
  No-grad path: dashed gray.
  Cross-panel reference (δ → Panel b): dashed, thin.
