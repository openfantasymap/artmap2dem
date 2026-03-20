# FOSS4G Talk Proposal

## Title

**From Ink to Elevation: Converting Artistic and Historical Maps into Digital Elevation Models with Open Source Tools**

---

## Abstract (Short, ~100 words)

Millions of artistic, historical, and fantasy maps contain rich implied topographic information — but no machine-readable elevation data. ArtMap2DEM is an open-source Python library that bridges this gap by applying computer vision and geomorphological simulation to automatically convert stylized, georeferenced maps into realistic Digital Elevation Models (DEMs). The pipeline extracts color, texture, edge, and pattern features; classifies terrain probabilistically; generates fractal-enhanced elevation grids; and enforces hydrological consistency. This talk demonstrates the approach, discusses design decisions, and explores use cases in historical cartography, game world modeling, and heritage digitization — all using standard FOSS geospatial tools.

---

## Abstract (Full, ~500 words)

Historical atlases, hand-painted cartographic artworks, and fictional world maps share a common challenge for the modern GIS practitioner: they encode spatial and topographic knowledge in purely visual form — through hachures, color conventions, brushstroke textures, and symbolic patterns — but offer no numeric elevation metadata. Converting these maps into quantitative geospatial datasets has traditionally required slow, expert-driven manual digitization.

**ArtMap2DEM** is a fully open-source Python library that automates this conversion through a five-stage processing pipeline, making the approach accessible to anyone in the FOSS4G community.

**Stage 1 — Feature Extraction**: The library uses OpenCV, scikit-image, and NumPy to extract a rich feature set from the input image, including HSV color analysis with K-means clustering, multi-scale Canny and Sobel edge detection, Hessian eigenvalue-based ridge detection, Local Binary Pattern texture descriptors, Gabor filter responses, and 2D FFT-based periodic pattern analysis.

**Stage 2 — Terrain Analysis**: Features are combined probabilistically to classify pixels into nine terrain categories (mountains, valleys, flat plains, water bodies, cliffs, etc.). Water bodies are identified through blue-hue and smooth-texture cues; peaks through texture complexity and local maxima; slopes through edge density gradients. Each pixel receives a confidence score per class.

**Stage 3 — DEM Generation**: Terrain probabilities drive elevation synthesis. Per-class elevation ranges are applied, then enriched with fractal Brownian motion noise (multi-octave, configurable persistence) to produce naturalistic terrain detail. Ridges and cliffs are selectively sharpened, while flat areas receive adaptive Gaussian smoothing. The result is a continuous 0–4000m (configurable) elevation raster.

**Stage 4 — Hydrological Processing**: A priority-flood algorithm eliminates spurious sinks. River channels are extracted by skeletonization and have monotonic downstream gradients enforced. D8 flow direction and Strahler stream order are computed. Water bodies are flattened per connected component.

**Stage 5 — GeoTIFF Output**: The final DEM is written as a georeferenced GeoTIFF via Rasterio, preserving the input CRS and affine transform. Utility functions provide hillshade, slope, aspect, curvature, and color-relief derivation.

The talk will include a live demonstration on a real historical map, benchmarks comparing the output against manual digitization, and a discussion of current limitations (occlusion, non-standard color conventions, ambiguous symbolism). We will also cover how the modular architecture enables contributors to swap in deep learning classifiers as a drop-in replacement for the rule-based terrain analyzer — a natural next step for the project.

**Use cases** span historical geography research, digital humanities, tabletop game asset pipelines, and heritage preservation, making ArtMap2DEM a project of broad interest across the FOSS4G community.

---

## Session Details

| Field              | Value                                                      |
|--------------------|------------------------------------------------------------|
| **Track**          | Geospatial Analysis / Remote Sensing / Cartography         |
| **Session Type**   | Standard presentation (25 minutes + 5 minutes Q&A)        |
| **Audience Level** | Intermediate (basic GIS and Python familiarity helpful)    |
| **License**        | MIT (fully open source)                                    |
| **Code URL**       | https://github.com/openfantasymap/artmap2dem                |

---

## Learning Outcomes

Attendees will leave with:

1. An understanding of how computer vision feature extraction can be applied to cartographic images to infer terrain characteristics.
2. Awareness of the key challenges (color ambiguity, non-standard symbology, georeferencing) and how ArtMap2DEM addresses them.
3. Knowledge of the hydrological post-processing steps needed to produce physically consistent DEMs.
4. A working mental model of the pipeline they can extend or adapt for their own historical/artistic map datasets.
5. Familiarity with the FOSS stack used: Rasterio, scikit-image, OpenCV, SciPy, NumPy, scikit-learn.

---

## Talk Outline (30 minutes total)

| Time     | Section                                                  |
|----------|----------------------------------------------------------|
| 0–3 min  | Motivation: the problem with artistic maps in GIS        |
| 3–7 min  | Pipeline overview and design philosophy                  |
| 7–12 min | Deep dive: feature extraction and terrain classification |
| 12–17 min| DEM generation: fractal noise, smoothing, scaling        |
| 17–21 min| Hydrological consistency enforcement                     |
| 21–25 min| Live demo on a historical map                            |
| 25–30 min| Limitations, roadmap (deep learning classifier), Q&A     |

---

## Speaker Bio (placeholder)

*The ArtMap2DEM project is developed under the OpenFantasyMap / OpenHistoryMap umbrella, bringing open-source geospatial tooling to artistic and historical cartographic digitization. The presenter has experience in GIS software development, cartographic analysis, and scientific Python, and has contributed to the FOSS geospatial ecosystem.*

---

## Why FOSS4G?

ArtMap2DEM is built exclusively on open-source components (Python, NumPy, SciPy, OpenCV, scikit-image, scikit-learn, Rasterio, Matplotlib) and is MIT licensed. It represents a novel application of existing FOSS geospatial infrastructure to a niche but growing need in digital humanities and historical GIS — precisely the kind of innovative, community-driven project FOSS4G exists to showcase.

---

## Supplementary Materials

- **Repository**: https://github.com/openfantasymap/artmap2dem
- **Example notebook**: `examples/basic_usage.py`, `examples/advanced_usage.py`
- **Output example**: `dem_results.png`, `output_dem.tif` (included in repo)
