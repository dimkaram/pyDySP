pyDySP: Dynamic Signal Processing in Python
===========================================

**pyDySP** is a lightweight Python package for **dynamic signal processing of
experimental data**, built around two core classes:

- ``Channel`` – a single time-history with rich metadata and lazy processing
- ``Test`` – a collection of Channels with batch tools, spectra, transfer
  functions, and plotting helpers

It is developed for laboratory and field measurements (shaking-table tests,
soil–structure interaction experiments, structural dynamics, etc.) but is
general enough for any time-series workflow.

Features
--------

- Drift correction, filtering, baseline correction and trimming
  (non-destructive, lazy processing)
- Fourier and Welch spectra with peak detection helpers
- Arias intensity and significant-duration windows
- Elastic response spectra (Newmark-beta method)
- Cross-spectra, coherence, transfer functions and time delays
- Batch processing on multi-channel tests (drift, filter, baseline, trim)
- Simple channel “health” diagnostics
- Campaigns of many tests: load a folder, select by number, tag or metadata,
  plot per-test values and transfer-function heatmaps across the campaign,
  and keep test information in an editable CSV
- SoFSI/EQUALS ``.mat`` and wide-format ``.csv`` I/O
- Publication-ready plotting utilities for notebooks and reports

Getting started
---------------

If you are new to pyDySP, the recommended reading order is:

1. :doc:`installation` – how to install pyDySP and set up an environment.
2. :doc:`quickstart` – short, end-to-end examples with real signals.
3. :doc:`pydysp` – API reference for ``Channel``, ``Test``, ``Campaign`` and helpers.

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   quickstart
   examples
   pydysp
