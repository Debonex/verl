SkipManager Usage Documentation
=================================

Last updated: 2026-04-20

Overview
--------

**SkipManager** (``verl.utils.skip.SkipManager``) is the **recommended** way to cache and reuse
rollout outputs in verl. It centralizes skip behavior under the top-level Hydra key ``skip``,
applies decorators by **role** (e.g. ``"rollout"``), and gates behavior on **trainer global steps**
listed in config.

.. important::

   The legacy utility ``verl.utils.rollout_skip.RolloutSkip`` (patching
   ``generate_sequences`` via ``actor_rollout_ref.rollout.skip``) is **deprecated** and will be
   removed after a compatibility window. New code and configs should use **SkipManager** and
   ``skip.rollout`` only. See :doc:`rollout_skip` for historical reference.

Relationship to legacy RolloutSkip
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Topic
     - Legacy (deprecated)
     - New (SkipManager)
   * - Python entry
     - ``RolloutSkip(config, wg).wrap_generate_sequences()``
     - ``SkipManager.init(config)`` + ``@SkipManager.annotate(role="rollout")``
   * - Hydra / config
     - ``actor_rollout_ref.rollout.skip.*``
     - ``skip.rollout.*`` (under ``SkipManagerConfig``)
   * - Step selection
     - ``max_dump_step`` / internal gen counters
     - Explicit ``skip.rollout.steps`` (trainer ``global_steps`` integers)

If **both** ``skip.rollout.enable`` and legacy ``actor_rollout_ref.rollout.skip.enable`` are true,
SkipManager emits a ``DeprecationWarning`` and **forces** the legacy flag to ``False`` so only one
mechanism runs.

When to use SkipManager
-----------------------

SkipManager is intended for the same scenarios as the legacy helper: **debugging**, **replay**,
and **fixed-experiment** runs where you want to avoid recomputing identical generations on selected
training steps.

1. Re-running experiments with the same configuration and cached tensors.
2. Speeding up iterations by skipping generation on chosen ``global_steps``.
3. Reproducing rollout tensors for a narrow set of steps.


Configuration (Hydra)
---------------------

Trainer YAML already reserves a ``skip`` block (see ``verl/trainer/config/ppo_trainer.yaml``).
Override fields as follows:

.. code-block:: bash

   skip.rollout.enable=True
   skip.rollout.dump_dir=/path/to/rollout_dump
   skip.rollout.steps='[1,2,3,10]'
   skip.rollout.action=cache

**Parameters (``skip.rollout`` / ``RolloutSkipConfig``)**

- **enable** (bool): Master switch for rollout skip handled by SkipManager.
- **dump_dir** (str): Root directory for cached ``DataProto`` shards (expanded with ``~``).
- **steps** (list[int]): Trainer **global step** values on which skip logic is *eligible* to run.
  Outside this set, decorated functions always execute normally.
- **action** (``cache`` \| ``repeat``): Strategy inside the registered rollout skip implementation.

.. note::

   Only ``cache`` and ``repeat`` are validated in ``RolloutSkipConfig`` today, even though
   ``SkipAction`` in code lists additional enum values for future modules.


On-disk layout (rollout role)
-----------------------------

Under ``dump_dir``, rollout skip uses a subdirectory keyed by experiment metadata and batch shape:

.. code-block:: text

    {dump_dir}/{experiment_name}_{project_name}/
        └── GBS{gbs}_N{n}_in{prompt_len}_out{response_len}/
            ├── {global_step}/
            │   ├── gen_batch.dp
            │   └── meta.json
            └── ...

This layout differs from the legacy ``genstep_000001/`` style documented in :doc:`rollout_skip`.
The new tree uses **integer directory names equal to trainer global step** when writing through
``prepare_data``.


Trainer integration status
---------------------------

- ``RayPPOTrainer`` initializes SkipManager early in ``fit()`` and keeps ``set_step`` aligned with
  ``global_steps``.
- Agent loop / rollout call sites can attach ``@SkipManager.annotate(role="rollout")`` where
  generation is performed.

Extend with custom skip modules
--------------------------------

1. Subclass ``BaseSkip`` from ``verl.utils.skip.base_skip``.
2. Decorate the class with ``@register_skip("your_role_name")``.
3. Add a matching field under ``SkipManagerConfig`` (or use structured defaults) so
   ``SkipManager.init`` can pass ``cls.config.get("your_role_name")`` into your class.

Further reading
---------------

- Legacy rollout skip (deprecated): :doc:`rollout_skip`
