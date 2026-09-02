"""Functions used by the small-signal-stability tutorial.

The notebook will contain explanations and short function calls.  The actual
ANDES operations stay here so they can be reused for both testing and dataset
generation. Shared system data are defined in ``system_constants.py``.
"""

from __future__ import annotations

import logging
import os
import platform
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from itertools import product
from pathlib import Path

import system_constants as const

# Set writable cache locations before importing ANDES and Matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str(const.CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(const.CACHE_DIR / "xdg"))

import andes
import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Warning-level logging avoids repeated ANDES setup messages in a large loop.
andes.config_logger(stream_level=30)


def get_case_path() -> Path:
    """Return the installed path of the configured ANDES case."""

    return Path(andes.get_case(const.CASE_NAME))


ENVIRONMENT_INFO = {
    "Python": platform.python_version(),
    "ANDES": andes.__version__,
    "NumPy": np.__version__,
    "pandas": pd.__version__,
    "Matplotlib": matplotlib.__version__,
    "Case": str(get_case_path()),
}


@dataclass(frozen=True)
class OperatingPoint:
    """UC quantities used to define one ANDES operating point.

    Tuple order is SG buses (2, 3, 6) and ``const.LOAD_BUSES``.
    All powers are in MW.

    The default is the stable, high-solar tutorial operating point.
    """

    sg_online: tuple[int, int, int] = (1, 1, 1)
    sg_power_mw: tuple[float, float, float] = (10.0, 10.0, 30.0)
    gfl_power_mw: float = 90.0
    load_power_mw: tuple[float, ...] = tuple(const.BASE_LOAD_POWER_MW)


@dataclass(frozen=True)
class SmallSignalResult:
    """Targets and one diagnostic returned by the ANDES calculation."""

    label: int  # 0 = stable; 1 = unstable or inside the stability margin
    critical_real: float  # regression target
    critical_imag: float
    slack_power_mw: float  # diagnostic only


@contextmanager
def silence_andes_output(enabled: bool = True):
    """Hide ANDES diagnostics during batch runs while preserving exceptions."""

    if not enabled:
        yield
        return

    previous_logging_disable = logging.root.manager.disable
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous_logging_disable)


def failure_category(error: Exception) -> str:
    """Return one summary category for errors with sample-specific values.
    all slack bus out of range errors are categorized as "Slack output is outside its limits."
    """

    message = str(error)
    if message.startswith("Slack output "):
        return "Slack output is outside its limits."
    return message


def load_system():
    """Load a fresh IEEE 14-bus system.

    A new system is required for each sample because ANDES changes model
    statuses and state variables in place.
    """

    const.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    system = andes.load(
        str(get_case_path()),
        no_output=True,
        pycode_path=str(const.CACHE_DIR / "andes_pycode"),
    )

    # Use the converter's 100-MVA rating and a deliberately weak intertie.
    system.PV.set(
        "pmax", const.GFL_PV_INDEX,
        const.GFL_POWER_MAX_MW / const.SYSTEM_BASE_MVA,
    )
    # The PLL remains at its original Kp=1.0 and Ki=0.2 values.
    # Bus the reactance at the GFL line is increased into a weak grid.
    system.Line.set(
        "x", const.WEAK_GRID_LINE_INDEX, const.WEAK_GRID_LINE_X
    )
    return system


def build_system(point: OperatingPoint):
    """Load the case and apply commitment, dispatch, solar and load values."""

    # These two checks prevent the most common bus-order mistakes.
    if len(point.sg_online) != 3 or len(point.sg_power_mw) != 3:
        raise ValueError("Provide three SG values in bus order (2, 3, 6).")
    if len(point.load_power_mw) != len(const.LOAD_BUSES):
        raise ValueError(f"Provide loads in bus order {const.LOAD_BUSES}.")

    system = load_system()

    # PV provides the power-flow set point; GENROU is the dynamic SG model.
    # Apply the commitment and dispatch of the three switchable SGs.
    for unit, online, power_mw in zip(
        const.SWITCHABLE_SGS, point.sg_online, point.sg_power_mw
    ):
        # For PV bus, set the bus (PV and GENROU) status, power set point, and voltage is fixed
        system.PV.set_status(unit["pv"], online)
        system.GENROU.set_status(unit["generator"], online)
        system.PV.set("p0", unit["pv"], power_mw / const.SYSTEM_BASE_MVA)

        if not online:
            # Avoid a positive mechanical-power minimum on an offline turbine.
            system.TGOV1.set("VMIN", unit["governor"], 0.0)

    # The GFL remains online; only its actual injection changes 
    # (available solar power - curtailed power). 
    # The voltage is fixed.
    system.PV.set("pmin", const.GFL_PV_INDEX, 0.0)
    system.PV.set(
        "p0", const.GFL_PV_INDEX,
        point.gfl_power_mw / const.SYSTEM_BASE_MVA,
    )

    # Apply the bus loads while preserving each bus's base Q/P ratio.
    base_q_over_p = system.PQ.q0.v / system.PQ.p0.v
    system.PQ.p0.v[:] = (
        np.asarray(point.load_power_mw) / const.SYSTEM_BASE_MVA
    )
    system.PQ.q0.v[:] = system.PQ.p0.v * base_q_over_p

    return system


def solve_power_flow(system) -> float:
    """Run AC power flow and return the realized slack output in MW."""

    if not system.PFlow.run():
        raise RuntimeError("ANDES power flow did not converge.")

    # The slack generator balances demand and AC losses.
    slack_power_mw = float(system.Slack.p.v[0] * const.SYSTEM_BASE_MVA)
    slack_min_mw = float(system.Slack.pmin.v[0] * const.SYSTEM_BASE_MVA)
    slack_max_mw = float(system.Slack.pmax.v[0] * const.SYSTEM_BASE_MVA)
    if not slack_min_mw <= slack_power_mw <= slack_max_mw:
        raise ValueError(
            f"Slack output {slack_power_mw:.2f} MW is outside "
            f"[{slack_min_mw:.2f}, {slack_max_mw:.2f}] MW."
        )

    # Apply the bus voltage limits already stored in the workbook.
    voltage = system.Bus.v.v
    if np.any(voltage < system.Bus.vmin.v) or np.any(voltage > system.Bus.vmax.v):
        raise ValueError("At least one bus voltage is outside its limits.")

    # The case has no branch ratings and its default Q limits are inconsistent
    # with the base case, so the introductory tutorial does not enforce them.
    return slack_power_mw


def calculate_small_signal(
    system,
    slack_power_mw: float,
    stability_margin: float = const.STABILITY_MARGIN,
) -> SmallSignalResult:
    """Initialize the dynamic models and calculate the physical eigenvalues."""

    # Initialize GENROU, governors, exciters, GFL controls and PLL from PF.
    system.TDS.init()
    if system.exit_code != 0 or not system.TDS.initialized:
        raise RuntimeError("ANDES dynamic initialization failed.")
    if not system.TDS.itm_step() or system.exit_code != 0:
        raise RuntimeError("ANDES could not evaluate the initialized DAE model.")

    # Form the reduced state matrix and compute its eigenvalues.
    state_matrix = system.EIG.calc_As()
    eigenvalues, _ = system.EIG.calc_eig(state_matrix)

    # Remove structural zero modes such as the reference-angle mode.
    physical_modes = eigenvalues[
        np.abs(eigenvalues) > system.EIG.config.tol
    ]
    if len(physical_modes) == 0:
        raise RuntimeError("No physical eigenvalues were found.")

    # The eigenvalue with the largest real part determines stability.
    critical = physical_modes[np.argmax(physical_modes.real)]
    result = SmallSignalResult(
        label=int(critical.real > -stability_margin),
        critical_real=float(critical.real),
        critical_imag=float(critical.imag),
        slack_power_mw=slack_power_mw,
    )
    return result


def sample_operating_points(
    samples_per_commitment: int = 2,
    random_seed: int = 7,
):
    """Yield random points for the seven nonzero SG commitments.

    The all-off commitment is excluded. Total demand varies, while every bus
    retains its base-load share so total load is a sufficient NN feature.
    """

    rng = np.random.default_rng(random_seed)
    for commitment in product((0, 1), repeat=3):
        if sum(commitment) == 0:
            continue

        for sample_index in range(samples_per_commitment):
            # Use one broad sample followed by three high-solar samples. This
            # keeps full-range coverage while adding more boundary information.
            boundary_sample = sample_index % 4 != 0

            # Resample until the approximate Slack output is inside its range.
            # Network losses are unknown before power flow, so the exact Slack
            # output is still checked later by solve_power_flow().
            while True:
                # Offline SGs have zero dispatch. Online SGs use their full
                # range for broad samples and low dispatch for high-solar ones.
                sg_power_values = []
                for online, pmin, pmax in zip(
                    commitment, const.SG_POWER_MIN_MW, const.SG_POWER_MAX_MW
                ):
                    if not online:
                        power_mw = 0.0
                    elif boundary_sample:
                        power_mw = float(
                            rng.uniform(pmin, min(pmin + 10.0, pmax))
                        )
                    else:
                        power_mw = float(rng.uniform(pmin, pmax))
                    sg_power_values.append(power_mw)
                sg_power = tuple(sg_power_values)

                # Broad samples cover the UC range; targeted samples improve
                # coverage close to the high-solar stability boundary.
                if boundary_sample:
                    solar_power = float(
                        rng.uniform(
                            const.TARGET_SOLAR_MIN_MW,
                            const.TARGET_SOLAR_MAX_MW,
                        )
                    )
                else:
                    solar_power = float(
                        rng.uniform(0.0, const.GFL_POWER_MAX_MW)
                    )

                # Sample only total demand. Every bus keeps its base-load share,
                # matching the simplified UC where spatial load distribution
                # is fixed and therefore is not an independent NN input.
                total_load_scale = rng.uniform(
                    const.LOAD_TOTAL_SCALE_MIN, const.LOAD_TOTAL_SCALE_MAX
                )
                load_values = const.BASE_LOAD_POWER_MW * total_load_scale
                load_power = tuple(float(power) for power in load_values)

                # Ignore network losses for this inexpensive pre-screen.
                approximate_slack_mw = (
                    sum(load_power) - sum(sg_power) - solar_power
                )
                if (
                    const.SLACK_POWER_MIN_MW
                    <= approximate_slack_mw
                    <= const.SLACK_POWER_MAX_MW
                ):
                    break

            yield OperatingPoint(
                sg_online=commitment,
                sg_power_mw=sg_power,
                gfl_power_mw=solar_power,
                load_power_mw=load_power,
            )


def generate_dataset(
    samples_per_commitment: int = 2,
    random_seed: int = 7,
    output_path: str | Path | None = (
        const.OUTPUT_DIR / "tutorial_small_signal_dataset.csv"
    ),
    verbose: bool = False,
    show_progress: bool = True,
    suppress_andes_output: bool = True,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Simulate sampled points and return the accepted dataset and failures.

    ``show_progress`` displays one progress bar. ANDES warnings from rejected
    samples are hidden by default. Set ``verbose=True`` only when detailed
    messages for every accepted or rejected sample are needed.
    """

    rows = []  # One row for each sample.
    failures: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()

    sampled_points = sample_operating_points(samples_per_commitment, random_seed)
    progress = tqdm(
        sampled_points,
        total=7 * samples_per_commitment,
        desc="ANDES samples",
        unit="sample",
        disable=not show_progress,
    )

    for number, point in enumerate(progress, start=1):
        try:
            with silence_andes_output(suppress_andes_output):
                system = build_system(point)
                slack_power_mw = solve_power_flow(system)
                result = calculate_small_signal(system, slack_power_mw)
        except (RuntimeError, ValueError) as error:
            failures[failure_category(error)] += 1
            if verbose:
                print(f"sample {number:04d}: rejected - {error}")
            progress.set_postfix(
                accepted=len(rows),
                rejected=sum(failures.values()),
                stable=label_counts[0],
                unstable=label_counts[1],
            )
            continue

        # Store scalar columns so the CSV is ready for neural-network training.
        row = {
            "u_2": point.sg_online[0],
            "u_3": point.sg_online[1],
            "u_6": point.sg_online[2],
            "P_g2_mw": point.sg_power_mw[0],
            "P_g3_mw": point.sg_power_mw[1],
            "P_g6_mw": point.sg_power_mw[2],
            "P_s8_mw": point.gfl_power_mw,
        }
        # The fixed bus shares allow the full ANDES load vector to be recovered
        # from this scalar total, so the redundant bus-level columns are omitted.
        row["P_load_total_mw"] = sum(point.load_power_mw)
        row.update(
            {
                "label": result.label,
                "critical_real": result.critical_real,
                "critical_imag": result.critical_imag,
                "slack_power_mw": result.slack_power_mw,
            }
        )
        rows.append(row)
        label_counts[result.label] += 1

        if verbose:
            print(
                f"sample {number:04d}: accepted - "
                f"label={result.label}, critical_real={result.critical_real:.6f}"
            )
        progress.set_postfix(
            accepted=len(rows),
            rejected=sum(failures.values()),
            stable=label_counts[0],
            unstable=label_counts[1],
        )

    if not rows:
        raise RuntimeError("No valid samples were generated.")

    dataset = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

    return dataset, failures
