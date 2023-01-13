This workflow's purpose was to validate `EchoNet <https://github.com/echonet/dynamic.git>`_ against novel echocardiograms.

Installation
------------
.. code-block:: bash

    git clone https://github.com/jsnng/echoanalysis-research.git
    cd echoanalysis-research
    python3 -m venv .
    source bin/activate
    cd echocardiogram-analysis-workflow/
    pip3 install -e .

Usage
-----
.. code-block:: bash

    python3 run.py