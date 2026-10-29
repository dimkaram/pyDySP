Installation
============

Stable release
--------------

pyDySP requires Python 3.11 or newer. Install the latest version from PyPI:

.. code-block:: bash

   pip install pydysp

(Recommended for most users.)

Install from source
-------------------

To install the development version directly from GitHub:

.. code-block:: bash

   git clone https://github.com/dkaramitros/pyDySP.git
   cd pyDySP
   pip install -e .

This installs pyDySP in *editable mode*, allowing you to modify the source and
immediately see the changes without re-installing.

Optional: Jupyter kernel
------------------------

If you want to run the example notebooks:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name pydysp-env

(You may choose any kernel name.)

Optional: sdypy (for modal analysis)
------------------------------------

The modal identification tools require the ``sdypy-EMA`` package:

.. code-block:: bash

   pip install sdypy-EMA

Environment management
----------------------

You may wish to use a virtual environment:

**venv**

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows
   pip install pydysp

**Conda / Mamba**

.. code-block:: bash

   mamba create -n pydysp python=3.11 pip
   mamba activate pydysp
   pip install pydysp
