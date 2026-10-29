Example Notebooks
=================

Two Jupyter notebooks in the ``examples/`` folder of the repository show
pyDySP at work on simulated data. Each can be read on GitHub, with its saved
outputs, or run in the browser on Binder without installing anything.

pyDySP Demo
-----------

A compact tour of the single-test features. It builds synthetic
multi-channel signals, wraps them in ``Channel`` and ``Test`` objects, and
covers processing (drift correction, filtering, trimming by Arias
intensity), plotting with peak annotations, batch processing and grid plots,
transfer functions, time delays, channel health checks, and the CSV round
trip with a channel-info file.

* `View on GitHub <https://github.com/dkaramitros/pyDySP/blob/main/examples/demo.ipynb>`__
* `Run on Binder <https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fdemo.ipynb>`__

pyDySP Campaign Demo
--------------------

A tour of the ``Campaign`` features on a simulated shaking-table campaign of
20 tests, in which an oscillator's natural frequency drops as damage
accumulates. It loads a folder of tests with descriptions and tags, selects
tests by tag and number, plots per-test values against test number,
accumulates residual displacements, overlays transfer functions, follows the
resonance with a heatmap and ``peak_frequency``, round-trips the test-info
CSV, and checks channels across tests.

* `View on GitHub <https://github.com/dkaramitros/pyDySP/blob/main/examples/campaign_demo.ipynb>`__
* `Run on Binder <https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fcampaign_demo.ipynb>`__
