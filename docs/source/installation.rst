Installation
============

Installing forestpd
-------------------

forestpd requires `Python <https://www.python.org/>`_ 3.10 or newer. Install it from PyPI:

.. code-block:: bash

   pip install forestpd

or, for development, as an editable install from a clone of the
`GitHub repository <https://github.com/nholtgrefe/forestpd>`_:

.. code-block:: bash

   git clone https://github.com/nholtgrefe/forestpd
   cd forestpd
   pip install -e .

Requirements
~~~~~~~~~~~~

pip installs the two required dependencies:

* `NumPy <https://numpy.org/>`_ >= 1.20, for the trees and the dynamic-programming tables.
* `Numba <https://numba.pydata.org/>`_ >= 0.56, which compiles the Newick parser, the cluster
  reduction and the dynamic program. Every compiled kernel has a NumPy/Python fallback with the
  same results. The fallback is used, with a :class:`~forestpd.ForestPDWarning` on import, if
  Numba cannot be imported, and is several times slower.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Extra
     - Contents
   * - ``phylozoo``
     - Conversion to and from `PhyloZoo <https://github.com/nholtgrefe/phylozoo>`_ trees, for
       plotting and the other PhyloZoo tools (see :ref:`manual-phylozoo`).
   * - ``maffe``
     - `maffe <https://github.com/roehrt/maffe>`_, for minimum agreement forests (see
       :ref:`installation-maffe`).
   * - ``dev``
     - pytest, black, ruff, mypy and PhyloZoo.
   * - ``docs``
     - Sphinx and the documentation theme.

.. code-block:: bash

   pip install "forestpd[phylozoo]"
   pip install "forestpd[maffe]"
   pip install -e ".[dev]"     # in a clone

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

In a clone:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html

The pages are then in ``docs/build/html/index.html``.

.. _installation-maffe:

Exact agreement forests with maffe
----------------------------------

The default agreement-forest stage, :class:`~forestpd.maf.HeuristicMAF`, needs no other software.
The exact stages :class:`~forestpd.maf.MaffeRMAF` and :class:`~forestpd.maf.MaffeUMAF`, and
:class:`~forestpd.maf.TieredMAF` (the heuristic, with an exact forest where it is large), need
`maffe <https://github.com/roehrt/maffe>`_, a C++ solver for minimum agreement forests with Python
bindings.

Installing maffe
~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "forestpd[maffe]"

This installs maffe from PyPI, which has ready-built packages for Python 3.11 to 3.14 on Linux and
Windows (x86_64). maffe needs Python >= 3.11.

forestpd works without maffe. maffe is only loaded when a :class:`~forestpd.maf.MaffeRMAF`,
:class:`~forestpd.maf.MaffeUMAF` or :class:`~forestpd.maf.TieredMAF` is created; without maffe,
this raises a :class:`~forestpd.ForestPDImportError` that gives the install command.

Building maffe from source
~~~~~~~~~~~~~~~~~~~~~~~~~~

On other platforms, such as macOS, Linux on ARM, or Linux with glibc older than 2.28, PyPI has no
package that fits, and maffe is built from source into the same Python environment:

.. code-block:: bash

   pip install "maffe @ git+https://github.com/roehrt/maffe@v0.1.1"

The build needs Python >= 3.11, a C++23 compiler (GCC >= 14 or a recent Clang), CMake, Ninja, Git
and ``patch``, and downloads and compiles SCIP and HiGHS. It takes about ten minutes. The commands
below pass build options with ``-Csetup-args``, which needs pip >= 23.1.

**Older compilers.** If the system compiler is too old (e.g. GCC 12 on Debian 12), use a GCC 14
toolchain from conda-forge. Link its C++ runtime statically, so that maffe keeps its own copy and
does not clash with other compiled packages:

.. code-block:: bash

   mamba create -p ./cxx14 -c conda-forge "gxx_linux-64=14" cmake ninja pkg-config patch
   export PATH="$PWD/cxx14/bin:$PATH"
   export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
   pip install "maffe @ git+https://github.com/roehrt/maffe@v0.1.1" \
     "-Csetup-args=-Dcpp_link_args=-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,-Bsymbolic"

On Debian and Ubuntu, conda's compiler does not search the system's multiarch headers, so the
build can fail with ``x86_64-linux-gnu/python3.11/pyconfig.h: No such file or directory``. Make
only that folder visible, then build again:

.. code-block:: bash

   mkdir -p pyinc/x86_64-linux-gnu
   ln -s /usr/include/x86_64-linux-gnu/python3.11 pyinc/x86_64-linux-gnu/python3.11
   export CPATH="$PWD/pyinc${CPATH:+:$CPATH}"

**Gurobi.** maffe can use Gurobi instead of HiGHS as SCIP's LP solver (build option ``gurobi``;
Gurobi needs a licence). With Gurobi in ``$GUROBI_HOME`` (e.g. ``.../gurobi1301/linux64``):

.. code-block:: bash

   mkdir -p grblib && ln -sf "$GUROBI_HOME/lib/libgurobi130.so" grblib/libgurobi.so
   export GUROBI_DIR="$GUROBI_HOME" CPATH="$GUROBI_HOME/include${CPATH:+:$CPATH}"
   export LIBRARY_PATH="$PWD/grblib:$GUROBI_HOME/lib"
   pip install --force-reinstall "maffe @ git+https://github.com/roehrt/maffe@v0.1.1" \
     -Csetup-args=-Dgurobi=enabled "-Csetup-args=-Dcpp_link_args=-Wl,-rpath,$GUROBI_HOME/lib"

Checking the installation
-------------------------

Test run
~~~~~~~~

Import forestpd, print the version and solve a small instance. The latest version is |version|.

.. code-block:: python

   import forestpd as fpd

   print(fpd.__version__)   # the installed version
   print(fpd.optimize_maxmpd(["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"], budget=2).value)
   # 8.0

To check that maffe is found:

.. code-block:: python

   from forestpd.maf import TieredMAF

   TieredMAF()   # raises ForestPDImportError without maffe

Troubleshooting
~~~~~~~~~~~~~~~

**The first call is slow.** Numba compiles the kernels the first time they run in a Python session
(a few seconds). Later calls in the same session are fast, and the compiled code is cached on disk
for later sessions. When timing forestpd, solve a small instance first.

**A ForestPDWarning on import.** Numba could not be imported, so forestpd uses its slower
fallbacks. Install ``numba`` (a regular dependency) into the same environment.

**A ForestPDWarning about rooted trees.** The trees look rooted (a ``[&R]`` comment, PhyloZoo
``DirectedPhyNetwork`` trees, or a binary root in every tree) while PD is unrooted, the default.
Pass ``rooted=True`` for rooted PD, or ignore the warning; see :ref:`manual-rooted`.
