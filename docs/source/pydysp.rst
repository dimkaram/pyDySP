API Reference
=============

This section documents the public API of **pyDySP**.

The core workflow is built around two main classes:

* :class:`pydysp.channel.Channel` – single time-history with metadata and lazy processing.
* :class:`pydysp.test.Test` – multi-channel experiment with batch tools, spectra,
  transfer functions, I/O and diagnostics.

A third class works across tests:

* :class:`pydysp.campaign.Campaign` – ordered series of tests from one
  experimental campaign, with selection, test-info CSV, channel checks, and
  results across tests (:class:`pydysp.result.CampaignResult`).

Additional modules provide spectral representations, Arias intensity, response spectra
and utilities such as downsampling.

.. toctree::
   :maxdepth: 2
   :caption: API:

   channel
   test
   campaign
   api_other
