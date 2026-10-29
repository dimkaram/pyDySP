# pyDySP  
Dynamic Signal Processing for Experimental Data

[![Tests](https://github.com/dkaramitros/pyDySP/actions/workflows/tests.yml/badge.svg)](https://github.com/dkaramitros/pyDySP/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/readthedocs/pydysp)](https://pydysp.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pydysp)](https://pypi.org/project/pyDySP/)
![Status](https://img.shields.io/pypi/status/pydysp)
![Maintenance](https://img.shields.io/maintenance/active/2025)
![License](https://img.shields.io/github/license/dkaramitros/pyDySP)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fdemo.ipynb)
[![Binder: Campaign demo](https://img.shields.io/badge/Binder-campaign%20demo-579aca)](https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fcampaign_demo.ipynb)

**pyDySP** is a lightweight Python library for **dynamic signal processing** of experimental data, developed for  
*the EQUALS and SoFSI Laboratories* at the *University of Bristol*.

It provides:

- A **Channel** class for time-history signals  
  (processing, spectra, trimming, Arias, response spectra, plots, metadata).
- A **Test** class for multi-channel experiments  
  (selection, batch processing, cross-spectra, transfer functions, modal analysis via *sdypy*, plotting utilities, `.mat`/`.csv` I/O).
- A **Campaign** class for a whole series of tests  
  (folder loading, selection by number/tag/metadata, per-test metrics, transfer-function overlays and heatmaps across tests, test-info CSV, sensor-change checks).
- Clean, readable, notebook-friendly code for research, teaching and reproducible workflows.
- Full internal documentation and example notebooks.

## Table of Contents

- [pyDySP](#pydysp)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Option A — Install from PyPI (recommended)](#option-a--install-from-pypi-recommended)
    - [Option B — Developer installation (editable)](#option-b--developer-installation-editable)
    - [Option C — Use directly from a folder](#option-c--use-directly-from-a-folder)
  - [Environment Setup](#environment-setup)
    - [Virtual environment (`venv`)](#virtual-environment-venv)
    - [Conda / Mamba environment](#conda--mamba-environment)
  - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
    - [Single channel](#single-channel)
    - [Multi-channel test](#multi-channel-test)
    - [Spectra \& trimming](#spectra--trimming)
    - [Response spectrum](#response-spectrum)
    - [Transfer functions](#transfer-functions)
    - [Channel health](#channel-health)
    - [Modal identification (optional)](#modal-identification-optional)
    - [Campaigns](#campaigns)
  - [Features](#features)
    - [Channel](#channel)
    - [Test](#test)
    - [Campaign](#campaign)
  - [Example Notebooks](#example-notebooks)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

### Option A — Install from PyPI (recommended)

```bash
pip install pydysp
```

For Jupyter notebooks:

```bash
pip install ipykernel
```

### Option B — Developer installation (editable)

```bash
git clone https://github.com/dkaramitros/pyDySP.git
cd pyDySP
pip install -e .
```

### Option C — Use directly from a folder

```python
import sys
sys.path.append("/path/to/parent/of/pydysp")
from pydysp import Channel, Test
```

## Environment Setup

### Virtual environment (`venv`)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install pydysp
```

### Conda / Mamba environment

```bash
conda create -n pydysp python=3.11 pip
conda activate pydysp
pip install pydysp
```

To create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
# replace with environment-dev.yml for development
conda activate pydysp
```

## Dependencies

Python 3.11 or newer. Required packages:

- numpy  
- scipy  
- matplotlib  
- tabulate  

Optional (for modal analysis):

```bash
pip install sdypy-EMA
```

## Quick Start

### Single channel

```python
from pydysp import Channel
import numpy as np

dt = 0.01
t = np.arange(0, 10, dt)
y = np.sin(2*np.pi*2*t)

ch = Channel(data=y, dt=dt, name_user="Acc1", quantity="acceleration", units="g")
ch.plot()
t_proc, y_proc = ch.processed()
```

### Multi-channel test

```python
from pydysp import Test

test = Test.from_channels(name="MyTest", channels=[ch1, ch2])
print(test.info())
test.plot_channels(ncols=2)
```

### Spectra & trimming

```python
spec = ch.fourier()
f_peak, s_peak = spec.peak()

ch_trim = ch.trim_by_arias()
```

### Response spectrum

```python
rs = ch.response_spectrum()
rs.plot(y="Sa", logx=True)
```

### Transfer functions

```python
f, H = test.transfer_function("Shaker", "Acc1", kind="H1")
```

### Channel health

```python
print(test.channel_health())
```

### Modal identification (optional)

```python
model = test.ema_model(input="Shaker", outputs=["Acc1","Acc2"], lower=2, upper=40)
model.get_poles()
model.select_poles()
model.print_modal_data()
```

### Campaigns

```python
from pydysp import Campaign, Test

camp = Campaign.from_folder(
    "data/shallow_foundation",              # *.mat, natural sort order
    loader=Test.from_sofsi_mat,
    number_from="SH2601_{number}_*",        # test numbers from file names
)
tests = camp.select(exclude_tags={"hammer"})

a_max = tests.metric("A1", "max_abs")       # signed, like Channel.max_abs()
ax = a_max.plot(highlight=21)               # value against test number

tf = tests.transfer_function("A8", "A14", nperseg=1024)
ax = tf.plot(kind="heatmap", fmax=40)       # test number vs frequency
tf.peak_frequency(fmin=5, fmax=40).plot(ax=ax, color="w")

camp.test_info_to_csv("tests.csv")          # edit, then reload with
camp = camp.with_test_info_from_csv("tests.csv")
```

Plot methods return the Matplotlib `Axes` they drew on (the figure is `ax.figure`).

## Features

### Channel
- Drift correction, filtering, baseline correction  
- Trimming (threshold, fraction, Arias)  
- Fourier & Welch spectra  
- Arias intensity  
- Response spectrum  
- Time-domain metrics  
- Metadata support  
- Plot utilities  

### Test
- Channel selection (by index, name, tags, slices)  
- Batch processing  
- Cross-spectra, transfer functions, delays  
- Modal analysis (via *sdypy*)  
- `.mat` (SoFSI/EQUALS) and `.csv` I/O  
- Grid plots & channel summary tables  
- Health diagnostics  

### Campaign
- Load a folder of tests (case-insensitive pattern, natural sort, test numbers from file names)  
- Select by number, name, tag or metadata; numbers stay with the tests  
- Per-test metrics (`max`, `min`, `max_abs`, `rms`, `residual`, or any function) with transforms and plots  
- Transfer functions and spectra across tests: overlay, heatmap, peak frequency  
- Test-info CSV (export, edit, reload) and channel checks for sensor changes  

## Example Notebooks

Located in `examples/`. Both use simulated data and run on Binder without installing anything.

- **[demo.ipynb](examples/demo.ipynb)** ([Binder](https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fdemo.ipynb)):
  a compact tour of `Channel` and `Test`: processing, plotting with peak annotations, Arias trimming,
  batch processing, transfer functions, time delays, channel health and the CSV round trip.
- **[campaign_demo.ipynb](examples/campaign_demo.ipynb)** ([Binder](https://mybinder.org/v2/gh/dkaramitros/pyDySP/HEAD?labpath=examples%2Fcampaign_demo.ipynb)):
  a simulated shaking-table campaign of 20 tests with `Campaign`: loading a folder, selection by tag
  and number, per-test metrics, transfer-function overlays and heatmaps that follow a drifting
  resonance, the test-info CSV and channel checks.

## Documentation

https://pydysp.readthedocs.io/en/latest/

## Contributing

Contributions are very welcome — whether bug reports, small fixes, feature ideas, or full pull requests.

To contribute:

1. **Fork** the repository on GitHub  
2. **Create a branch** for your change  
3. **Commit** your edits with clear messages  
4. **Run the tests** locally:

   ```bash
   pip install -e .
   pip install pytest
   pytest -q
   ```

5. **Open a Pull Request** describing the change and its motivation

Please keep code style readable and consistent with the existing project  
(docstrings, small focused functions, and meaningful error messages).

If you prefer to discuss an idea before writing code, feel free to open a **GitHub Issue**.

## License

MIT License
