.. image:: _static/forestpd-logo.svg
   :alt: ForestPD logo
   :width: 420px
   :align: left
   :class: only-light

.. image:: _static/forestpd-logo-dark.svg
   :alt: ForestPD logo
   :width: 420px
   :align: left
   :class: only-dark

.. raw:: html

   <div style="clear: both;"></div>
   <br>

ForestPD (Python package ``forestpd``) selects a set of taxa that maximizes **phylogenetic
diversity (PD) summed over many phylogenetic trees**, such as a Bayesian posterior sample,
bootstrap replicates or gene trees, under a budget on taxon costs and optionally with extinction
risks. It works on the trees directly, without a summary tree or network.

The algorithm is exact. Its running time is exponential only in the size of the agreement forests
of the clusters that all trees share, so it handles thousands of trees on tens of thousands of
taxa when the trees mostly agree. One run gives the optimum for every budget up to the one
requested.

.. code-block:: python

   import forestpd as fpd

   result = fpd.optimize_maxmpd(
       ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"],  # or a file of Newick trees
       budget=2,
       return_taxa=True,
   )
   result.value   # 8.0
   result.taxa    # {'A', 'D'} or {'B', 'C'}: both are optimal

Contents
--------

:doc:`installation` covers forestpd and the optional maffe package. :doc:`problem` defines the
problem and outlines the algorithm. The :doc:`manual` describes how to use forestpd, and
:doc:`examples` has a complete analysis. The :doc:`api/index` documents every public function and
class, and the :doc:`changelog` lists the changes per version.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   problem
   manual
   examples
   api/index
   changelog

Indices
-------

* :ref:`genindex`: all functions and classes.
* :ref:`modindex`: all modules.
* :ref:`search`: search the documentation.

Citation
--------

If you use ForestPD in your research, please cite:

   Niels Holtgrefe, Jannik Schestag, and Norbert Zeh.
   *ForestPD: Maximizing Phylogenetic Diversity under Phylogenetic and Extinction Uncertainty.*
   2026.
