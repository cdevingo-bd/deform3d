# Contributing

Use a separate virtual environment and install `python -m pip install -e '.[dev]'`.
Run the C++ and default Python checks in [docs/testing.md](docs/testing.md).
Run the relevant host/GPU checks if changing those adapters.

Keep the C++ core independent of Python, MuJoCo, Warp, and task frameworks.
Keep solver changes opt-in and version guarded. Add regression coverage for
changes to binding formats, state refresh, force transfer, or numerical kernels.
Do not relax numerical tolerances to make a backend comparison pass.

For performance changes, report the hardware, package versions, world count,
physics settings, warmup, and measured scope. Separate physics, deformation,
transfers, and rendering. Include numerical error and capacity failures.

Build products, JIT caches, recordings, and generated reports belong in ignored
directories. Only contribute assets you have permission to redistribute under
an explicitly identified license.
