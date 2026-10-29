Campaign API
============

A :class:`~pydysp.campaign.Campaign` holds an ordered series of
:class:`~pydysp.test.Test` objects from one experimental campaign. Each test
keeps its own **test number** through every selection; the 0-based **index**
is its position in the campaign currently held.

Analyses across tests return a :class:`~pydysp.result.CampaignResult`: one
value per test (:meth:`~pydysp.campaign.Campaign.metric`) or one curve per
test (:meth:`~pydysp.campaign.Campaign.transfer_function`,
:meth:`~pydysp.campaign.Campaign.spectrum`).

Conventions
-----------

* **Signed metrics.** ``"max"``, ``"min"``, ``"max_abs"`` and ``"residual"``
  are signed. ``"max_abs"`` is the value of the sample with the largest
  magnitude, keeping its sign, exactly like :meth:`Channel.max_abs()
  <pydysp.channel.Channel.max_abs>`; call ``.abs()`` on the result for the
  magnitude.
* **Residual window.** ``"residual"`` is the mean of the final ``n_last``
  samples (default 50). ``n_last`` is a number of samples, not seconds.
* **Plots return the Axes.** Every ``plot`` method draws on one set of axes
  and returns those axes. Pass ``ax=`` to draw into your own subplot; the
  figure is ``ax.figure``. No figure size, font or colour scheme is set, so
  your Matplotlib style applies.
* **Missing channels.** A test without the requested channel gives NaN (or no
  curve) and a warning; ``strict=True`` raises :class:`KeyError` instead.
  Refer to channels by name, not by integer position.

Test-info CSV
-------------

:meth:`~pydysp.campaign.Campaign.test_info_to_csv` writes one row per test,
and :meth:`~pydysp.campaign.Campaign.with_test_info_from_csv` reads it back
into a new campaign. Example file:

.. code-block:: text

    number,name,description,tags,amp
    1,Test 001,Hammer Test: Foundation,hammer,
    3,Test 003,White Noise 0.06g,white-noise,0.06
    4,Test 004,Record 901,"record,main",0.20

* ``number`` and ``name`` are always written. ``description``, ``tags``,
  ``timestamp``, ``source_file`` and one column per simple ``Test.meta`` entry
  (numbers and text of at most 80 characters) are written when at least one
  test has a value. Bulky metadata, such as the SoFSI file header, is skipped.
* Tags are comma-separated in one cell.
* On reading, rows are matched to tests by ``number``, or by ``name`` if there
  is no ``number`` column. Headers ignore case, spaces and underscores, and
  common aliases are accepted (for example ``No`` or ``Test number``).
* Extra columns go into ``Test.meta``: numbers if every filled cell is a
  number, text otherwise. Empty cells leave the test unchanged, and
  ``source_file`` is not read back. A row that matches no test raises
  :class:`ValueError`.

Campaign
--------

.. automodule:: pydysp.campaign
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

.. automodule:: pydysp.result
   :members:
   :undoc-members:
   :show-inheritance:
