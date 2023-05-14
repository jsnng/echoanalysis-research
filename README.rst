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

:code:`run.py` contains an example on usage. Unprocessed videos are placed in a child directory :code:`Videos`.
After running :code:`run.py`, processed videos are in :code:`.npz` uncompressed archives. Segmentation 
predictions are in saved :code:`.npy` arrays. These files are saved in the current working directory

.. code-block:: bash

    python3 run.py