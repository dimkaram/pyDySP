Quick Start
===========

Creating a ``Channel``
----------------------

A ``Channel`` represents a single time-history with metadata and lazy
processing (drift, filter, baseline, trimming):

.. code-block:: python

    from pydysp.channel import Channel
    import numpy as np

    dt = 0.01
    t = np.arange(0, 10, dt)
    y = np.sin(2 * np.pi * 2 * t)

    ch = Channel(
        data=y,
        dt=dt,
        name_user="Acc1",
        quantity="acceleration",
        units="g",
    )

    # Apply processing (non-destructive)
    ch2 = (
        ch.drift_corrected(points=100)
          .filtered(fc=20)
          .baseline_corrected()
    )

    # Plot processed time-history
    ch2.plot()

Creating a ``Test`` (multi-channel experiment)
----------------------------------------------

A ``Test`` stores and manages multiple channels, with tools for batch
processing, plotting, spectra, transfer functions, and more:

.. code-block:: python

    from pydysp.test import Test

    test = Test.from_channels(
        name="MyTest",
        channels=[ch1, ch2],
    )

    # Plot channels in a grid
    test.plot_channels(ncols=2)

    # Compute transfer function H1
    f, H = test.transfer_function("Acc1", "Acc2", kind="H1")

Working with a ``Campaign`` (many tests)
----------------------------------------

A ``Campaign`` holds the tests of one experimental campaign in order. Every
test keeps its own **test number** (the one people say aloud) through every
selection, and plots use it as the x-axis.

Loading a folder
~~~~~~~~~~~~~~~~

``from_folder`` matches file names ignoring case (``*.mat`` also finds
``.MAT``) and sorts them in natural order (``Test_2`` before ``Test_10``).
Test numbers can be read from the file names, and descriptions, tags and
metadata are given as lists in the same order:

.. code-block:: python

    from pydysp import Campaign, Test

    camp = Campaign.from_folder(
        "data/shallow_foundation",
        pattern="*.mat",
        loader=Test.from_sofsi_mat,           # any callable: path -> Test
        number_from="SH2601_{number}_*",      # matched against the file name
        name_template="Test {number:03d}",
        descriptions=["Hammer", "White noise 0.06g", "Record 901"],
        tags=[{"hammer"}, {"white-noise"}, {"record"}],
        meta={"amp": [None, 0.06, 0.20]},     # columns, one value per file
    )
    print(camp.info())

The ``loader`` can be a whole processing chain, for example
``lambda f: Test.from_sofsi_mat(f).drift_corrected()``. Without
``number_from`` or ``numbers=``, tests are numbered 1, 2, 3, ...

Selecting tests
~~~~~~~~~~~~~~~

Selections return a new campaign; tests keep their numbers:

.. code-block:: python

    tests = camp.select(exclude_tags={"hammer"})     # skip hammer tests
    wn = camp.select(tags={"white-noise"})
    late = camp.select(numbers=range(29, 40))        # gaps are fine
    strong = camp.select(where=lambda t: t.meta.get("amp", 0) > 0.1)

    test21 = camp.get(number=21)                     # look up by number
    first = camp[0]                                  # or by position

Values across tests
~~~~~~~~~~~~~~~~~~~

``metric`` reduces one channel in each test to one number and returns a
``CampaignResult``:

.. code-block:: python

    a_max = tests.metric("A1", "max_abs")   # signed peak, like Channel.max_abs()
    a_rms = tests.metric("A1", "rms")
    d_res = tests.metric("L1", "residual", n_last=50)   # mean of last 50 samples

    ax = a_max.plot(highlight=21)           # value against test number
    d_res.abs().cumsum().plot()             # accumulated residual displacement

    f_max = tests.metric("F1", "max_abs")
    f_max.plot(x=d_res, origin=True)        # force against displacement

    print(a_max.to_table())
    a_max.to_csv("a1_max.csv")

Built-in metrics are ``"max"``, ``"min"``, ``"max_abs"``, ``"rms"`` and
``"residual"``, and any function of a ``Channel`` that returns a number also
works. All are signed: ``"max_abs"`` is the value of the sample with the
largest magnitude, keeping its sign, exactly like ``Channel.max_abs()``. Use
``.abs()`` for the magnitude. ``n_last`` is a number of samples, not seconds.

A test without the channel gives NaN and a warning (``strict=True`` raises an
error instead), so a replaced sensor shows as a gap rather than a crash.

Every ``plot`` method returns the Axes it drew on, so results can be placed
in your own figure with ``ax=``; the figure is ``ax.figure``:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, layout="constrained")
    a_max.plot(ax=axs[0])
    a_rms.plot(ax=axs[1], ylabel="A1 RMS [g]")

Transfer functions and spectra across tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``transfer_function`` and ``spectrum`` return one curve per test:

.. code-block:: python

    wn = camp.select(tags={"white-noise"})
    tf = wn.transfer_function("A8", "A14", kind="H1", nperseg=1024)

    tf.plot(fmax=50)                                # overlay, labelled "Test n"
    camp.transfer_function("A8", "A14", tests=[3, 29], nperseg=1024).plot()

    psd = wn.spectrum("A14", method="welch", nperseg=1024)

The **heatmap** shows how a resonance moves through the campaign. It is laid
out like a spectrogram: test number along the horizontal axis, increasing to
the right, and frequency along the vertical axis, increasing upward. Colour
shows the amplitude. Each column is centred on its test number, so a skipped
number gives a wider column. Because the x-axis is the test number,
``peak_frequency`` can be drawn on top:

.. code-block:: python

    from matplotlib.colors import LogNorm

    ax = tf.plot(kind="heatmap", fmin=5, fmax=40)            # optional: cmap=, norm=
    f_peak = tf.peak_frequency(fmin=5, fmax=40)              # a scalar result in Hz
    f_peak.plot(ax=ax, color="w", ylabel="Frequency [Hz]")

    tf.plot(kind="heatmap", fmax=40, norm=LogNorm())         # log colour scale

The heatmap works best on tests with the same input (for example white
noise), so that any change comes from the system. It needs all tests to share
one frequency grid (same sampling interval and ``nperseg``) and raises an
error otherwise; curves are never interpolated. Transfer functions are
complex: plots and ``to_csv`` show the magnitude by default, and
``part="phase"`` (or ``"real"``, ``"imag"``) is available.

Test-info CSV
~~~~~~~~~~~~~

Define the descriptions and tags once, export them, and from then on load the
files and read the CSV back:

.. code-block:: python

    camp.test_info_to_csv("tests.csv")

    camp = Campaign.from_folder(
        "data/shallow_foundation", number_from="SH2601_{number}_*"
    ).with_test_info_from_csv("tests.csv")

The file has one row per test. ``number`` and ``name`` are always written;
other columns appear when at least one test has a value. Tags are
comma-separated, and extra columns become ``Test.meta`` entries (numbers if
every filled cell is a number):

.. code-block:: text

    number,name,description,tags,amp
    1,Test 001,Hammer Test: Foundation,hammer,
    3,Test 003,White Noise 0.06g,white-noise,0.06
    4,Test 004,Record 901,"record,main",0.20

Rows are matched to tests by ``number`` (or by ``name`` if there is no
``number`` column), and empty cells leave a test unchanged.

Sensor changes
~~~~~~~~~~~~~~

If a sensor is replaced mid-campaign, give the new channel the same
``name_user`` in a second channel-info CSV and apply each file to its range
of tests. Then check that every channel exists in every test:

.. code-block:: python

    camp = camp.with_channel_info_from_csv("channels_a.csv", numbers=range(1, 15))
    camp = camp.with_channel_info_from_csv("channels_b.csv", numbers=range(15, 36))
    print(camp.check_channels())
