Simulation
==========

``ExperimentConfig`` requires a seed and names the independent dimensions as
``n_replicates`` and ``horizon``. Common reward coupling is the default paired
design; independent policy reward streams are available explicitly.

Results contain read-only arrays and recursively immutable configuration,
metadata, and provenance. Use ``SimulationResult.equals`` instead of dataclass
``==`` for value comparison. JSON and NPZ save methods are atomic, normalize a
missing suffix, and return the actual destination path.

Contextual runs record a digest by default. Set ``record_contexts=True`` to keep
the complete context tensor when downstream auditing requires it.

.. automodule:: pymab.simulation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pymab.metrics
   :members:
   :undoc-members:
   :show-inheritance:
