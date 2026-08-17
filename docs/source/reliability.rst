Reliability contracts
=====================

PyMAB fails explicitly when input data, component capabilities, logged-data
support, or persisted results violate their contracts. Catch the most specific
exception that your application can recover from; ``PyMABError`` is the common
package-level base.

Run provenance
--------------

Every ``SimulationResult`` records the PyMAB, Python, and NumPy versions, the
versioned random-stream scheme, and immutable snapshots of the configured
environment and policies. These snapshots contain constructor configuration,
not learned policy state. A migrated legacy result uses explicit ``"unknown"``
values instead of inventing provenance.

Configuration and metadata are recursively immutable and JSON-compatible.
Arrays are read-only. Use ``SimulationResult.equals`` for value comparison;
mutable policies and environments intentionally retain identity equality.

Persistence
-----------

JSON and compressed NPZ writes use a sibling temporary file, flush it, and
atomically replace the destination. A missing ``.json`` or ``.npz`` suffix is
added consistently during save and load. Wrong suffixes, malformed files,
unsupported schemas, and invalid fields raise ``SerializationError``.

Contextual experiments always record a deterministic context digest. Set
``ExperimentConfig(record_contexts=True)`` only when the complete
``(replicate, step, arm, feature)`` context tensor is needed; it can be large.

.. automodule:: pymab.errors
   :members:
   :show-inheritance:

.. automodule:: pymab.provenance
   :members:
   :show-inheritance:

.. automodule:: pymab.persistence
   :members:
   :show-inheritance:
