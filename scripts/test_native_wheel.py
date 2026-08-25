"""Install one wheel outside the source tree and smoke-test native execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

SMOKE = """
import numpy as np
from pymab import _native
from pymab.environments import BanditEnvironment
from pymab.policies import UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig

assert _native.native_available()
policy = UCBPolicy(n_arms=2)
assert policy.backend == "rust"
result = Experiment(
    environment=BanditEnvironment(means=np.array([0.0, 1.0])),
    policies={"ucb": policy},
    config=ExperimentConfig(horizon=3, n_replicates=2, seed=7, backend="rust"),
).run()
assert result.rewards.shape == (2, 3, 1)
assert result.provenance.backend == "rust"
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    arguments = parser.parse_args(argv)
    wheels = sorted(arguments.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    with tempfile.TemporaryDirectory(prefix="pymab-wheel-test-") as directory:
        environment = Path(directory) / "venv"
        subprocess.run(  # noqa: S603 - current interpreter and private temp path
            [sys.executable, "-m", "venv", str(environment)], check=True
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(  # noqa: S603 - interpreter is the venv created above
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheels[0]),
            ],
            check=True,
        )
        subprocess.run(  # noqa: S603 - fixed smoke program and private interpreter
            [str(python), "-c", SMOKE], cwd=directory, check=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
