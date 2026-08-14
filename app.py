from __future__ import annotations

import io
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib import rcParams

import problems as problem_api
from solver import solve_problem


# The reviewed problem builders use ``force_amplitude`` while the historical
# public compatibility adapter still emits ``C``. Keep that compatibility
# adjustment local to the Streamlit process so the scientific modules and
# experiment artifacts remain untouched.
def _install_streamlit_problem_compatibility() -> None:
    current_adapter = problem_api._oscillator_params
    if getattr(current_adapter, "_streamlit_force_amplitude_compat", False):
        return

    def compatible_adapter(param: Mapping[str, float]) -> dict[str, float]:
        converted = current_adapter(param)
        if "force_amplitude" not in converted and "C" in converted:
            converted = {**converted, "force_amplitude": converted["C"]}
        return converted

    compatible_adapter._streamlit_force_amplitude_compat = True  # type: ignore[attr-defined]
    problem_api._oscillator_params = compatible_adapter


_install_streamlit_problem_compatibility()


rcParams.update(
    {
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "font.size": 20,
        "axes.labelsize": 25,
        "legend.fontsize": 20,
    }
)

MEMORY_WARNING_GIB = 1.0
MEMORY_BLOCK_GIB = 2.0

PROBLEM_MAP = {
    "Forced Harmonic Oscillator": "OAF",
    "Damped Harmonic Oscillator": "OAA",
    "2D Heat Equation": "C2D",
}

# Defaults are grounded in the reviewed parameter sweep. OAF fits the hosted
# memory budget at the exact selected benchmark. For OAA and C2D, use the best
# sampled point that remains below the app's 2 GiB core-tensor guard.
DEFAULT_TAU = {"OAF": 3967, "OAA": 5395, "C2D": 23}
DEFAULT_MU = {"OAF": 2000, "OAA": 2000, "C2D": 512}

REVIEWED_BENCHMARK = {
    "OAF": {"mu": 2000, "tau": 3966.6280166708093},
    "OAA": {"mu": 4096, "tau": 8892.098602882095},
    "C2D": {"mu": 1024, "tau": 44.97941097041009},
}

DEFAULT_PARAMS = {
    "OAF": {
        "k": 5.0,
        "m": 7.0,
        "nu": 0.4,
        "C": 9.0,
        "x0": 5.0,
        "xq": 3.0,
        "dt": 0.5,
        "steps": 100,
    },
    "OAA": {
        "k": 5.0,
        "m": 7.0,
        "nu": 0.4,
        "C": 9.0,
        "x0": 5.0,
        "xq": 3.0,
        "dt": 0.5,
        "steps": 100,
        "gamma": 0.1,
    },
    "C2D": {
        "k": 3.0,
        "u1x": 5.0,
        "u2x": 3.0,
        "u1y": 4.0,
        "u2y": 2.0,
        "dxy": 0.5,
        "nx": 20,
        "ny": 20,
    },
}

PARAM_DESCRIPTIONS = {
    "OAF": {
        "k": "Spring constant (N/m)",
        "m": "Mass (kg)",
        "nu": "Driving frequency parameter nu",
        "C": "Driving force amplitude",
        "x0": "Left boundary x(0)",
        "xq": "Right boundary x(T)",
        "dt": "Time step (s)",
        "steps": "Number of time intervals",
    },
    "OAA": {
        "k": "Spring constant (N/m)",
        "m": "Mass (kg)",
        "nu": "Driving frequency parameter nu",
        "C": "Driving force amplitude",
        "x0": "Left boundary x(0)",
        "xq": "Right boundary x(T)",
        "dt": "Time step (s)",
        "steps": "Number of time intervals",
        "gamma": "Damping coefficient",
    },
    "C2D": {
        "k": "Thermal conductivity",
        "u1x": "Boundary temperature at x side 1",
        "u2x": "Boundary temperature at x side 2",
        "u1y": "Boundary temperature at y side 1",
        "u2y": "Boundary temperature at y side 2",
        "dxy": "Grid spacing",
        "nx": "Number of interior x grid points",
        "ny": "Number of interior y grid points",
    },
}


def _as_numpy(value: np.ndarray | torch.Tensor | list[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _hhl_dimension(problem: str, params: Mapping[str, float]) -> int:
    if problem == "OAF":
        return int(params["steps"]) - 1
    if problem == "OAA":
        return 2 * (int(params["steps"]) - 1)
    if problem == "C2D":
        return int(params["nx"]) * int(params["ny"])
    raise ValueError(f"Unknown problem type: {problem}")


def _estimated_core_storage_gib(problem: str, params: Mapping[str, float], mu: int) -> float:
    """Lower-bound estimate for the largest explicitly materialized TN tensors."""
    n = _hhl_dimension(problem, params)
    complex128_bytes = 16
    estimated_bytes = complex128_bytes * (
        mu * n * n
        + 3 * mu * mu
        + mu * n
        + 2 * n * n
    )
    return estimated_bytes / (1024**3)


def _validate_inputs(
    problem: str,
    params: Mapping[str, float],
    mu: int,
    tau: float,
) -> list[str]:
    errors: list[str] = []
    if mu < 3:
        errors.append("Phase-register dimension mu must be at least 3.")
    if tau <= 0:
        errors.append("Spectral parameter tau must be positive.")

    if problem in {"OAF", "OAA"}:
        if int(params["steps"]) < 2:
            errors.append("The oscillator requires at least two time intervals.")
        if float(params["dt"]) <= 0:
            errors.append("The time step dt must be positive.")
        if float(params["m"]) == 0:
            errors.append("The mass m must be non-zero.")
    elif problem == "C2D":
        if int(params["nx"]) < 2 or int(params["ny"]) < 2:
            errors.append("The 2D grid requires nx >= 2 and ny >= 2.")
        if float(params["dxy"]) <= 0:
            errors.append("The grid spacing dxy must be positive.")
        if float(params["k"]) == 0:
            errors.append("Thermal conductivity k must be non-zero.")

    return errors


def _heatmap_with_boundaries(
    algorithm_result: np.ndarray,
    params: Mapping[str, float],
) -> np.ndarray:
    """Build a rectangular plotting field without changing the C2D solve."""
    nx = int(params["nx"])
    ny = int(params["ny"])
    interior = np.asarray(algorithm_result).reshape(nx, ny)
    field = np.empty((nx + 2, ny + 2), dtype=interior.dtype)
    field[1:-1, 1:-1] = interior
    field[0, :] = params["u1x"]
    field[-1, :] = params["u2x"]
    field[1:-1, 0] = params["u1y"]
    field[1:-1, -1] = params["u2y"]
    return field


def _figure_png(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


st.set_page_config(
    page_title="HHL spectral filter with tensor networks",
    page_icon="🧮",
    layout="wide",
)

st.markdown(
    """
    <meta property="og:title" content="Finite-resolution HHL spectral filter with tensor networks">
    <meta property="og:description" content="Interactive tensor-network simulation of the finite-resolution HHL spectral filter">
    <meta property="og:image" content="https://raw.githubusercontent.com/DOKOS-TAYOS/Tensor_Networks_HHL_algorithm/main/thumbnail.png">
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.title("HHL spectral filter with tensor networks")
    st.markdown(
        """
        This is the interactive front end for the tensor-network implementation of
        the finite-resolution HHL spectral filter described in
        [Simulating the finite-resolution HHL spectral filter with tensor networks and qudits](https://arxiv.org/abs/2309.05290).

        The manuscript reproducibility experiments and their generated artifacts are
        separate from this resource-constrained interactive demo. The application
        calls the repository's existing scientific solver without modifying its
        numerical implementation.

        Code developed by [Alejandro Mata Ali](https://github.com/DOKOS-TAYOS/Tensor_Networks_HHL_algorithm).
        """
    )

    problem_selection = st.selectbox(
        "Select the problem to solve",
        list(PROBLEM_MAP),
        help="Choose one of the three application examples.",
    )
    problem = PROBLEM_MAP[problem_selection]

    parameter_col, phase_col = st.columns(2)
    with parameter_col:
        default_tau = DEFAULT_TAU[problem]
        tau = st.slider(
            "Spectral parameter tau (API name t)",
            min_value=max(1, int(default_tau / 10)),
            max_value=int(default_tau * 10),
            value=default_tau,
            key=f"tau_{problem}",
            help=(
                "Finite spectral-grid spacing is Delta lambda = 1/tau. "
                f"Reviewed benchmark tau: {REVIEWED_BENCHMARK[problem]['tau']:.2f}."
            ),
        )

    with phase_col:
        num_eigen = st.slider(
            "Phase-register dimension mu (API name num_eigen)",
            min_value=100,
            max_value=4096,
            value=DEFAULT_MU[problem],
            key=f"mu_{problem}",
            help=(
                "mu is the phase-register dimension, not a count of eigenvalues. "
                f"Reviewed benchmark mu: {REVIEWED_BENCHMARK[problem]['mu']}."
            ),
        )

with right_col:
    st.subheader("Problem parameters")
    params: dict[str, float] = {}
    input_columns = st.columns(4)
    for index, (key, default_value) in enumerate(DEFAULT_PARAMS[problem].items()):
        with input_columns[index % 4]:
            if key in {"steps", "nx", "ny"}:
                params[key] = st.number_input(
                    PARAM_DESCRIPTIONS[problem][key],
                    value=int(default_value),
                    step=1,
                    key=f"{problem}_{key}",
                    help=f"Parameter {key} for the {problem} example.",
                )
            else:
                params[key] = st.number_input(
                    PARAM_DESCRIPTIONS[problem][key],
                    value=float(default_value),
                    step=0.1,
                    key=f"{problem}_{key}",
                    help=f"Parameter {key} for the {problem} example.",
                )

    benchmark = REVIEWED_BENCHMARK[problem]
    st.caption(
        f"Hosted default: mu={DEFAULT_MU[problem]}, tau={DEFAULT_TAU[problem]}. "
        f"Reviewed benchmark: mu={benchmark['mu']}, tau={benchmark['tau']:.2f}."
    )

    estimated_storage_gib = _estimated_core_storage_gib(problem, params, num_eigen)
    st.caption(
        f"Approximate lower-bound storage for explicitly materialized TN tensors: "
        f"{estimated_storage_gib:.2f} GiB. Temporary PyTorch allocations are not included."
    )
    if estimated_storage_gib >= MEMORY_WARNING_GIB:
        st.warning(
            "This configuration is memory-intensive for a hosted Streamlit process. "
            "Reducing mu or the discretization size can improve reliability."
        )

    run_solver = st.button("Run solver", type="primary")

if run_solver:
    validation_errors = _validate_inputs(problem, params, num_eigen, tau)
    if validation_errors:
        for message in validation_errors:
            st.error(message)
    elif estimated_storage_gib > MEMORY_BLOCK_GIB:
        st.error(
            f"This configuration has an estimated core tensor footprint of "
            f"{estimated_storage_gib:.2f} GiB before temporary allocations. "
            "Lower mu or the problem discretization before running it in this hosted app."
        )
    else:
        with st.spinner("Solving the problem..."):
            try:
                algorithm_result, actual_result, x_axis, _ = solve_problem(
                    problem=problem,
                    params=params,
                    num_eigen=num_eigen,
                    t=tau,
                )
                algorithm_np = _as_numpy(algorithm_result)
                actual_np = _as_numpy(actual_result)
                x_axis_np = np.asarray(x_axis)
                result_2d = (
                    _heatmap_with_boundaries(algorithm_np, params)
                    if problem == "C2D"
                    else None
                )
                st.session_state["last_solver_result"] = {
                    "problem": problem,
                    "problem_selection": problem_selection,
                    "params": dict(params),
                    "mu": int(num_eigen),
                    "tau": float(tau),
                    "algorithm_result": algorithm_np,
                    "actual_result": actual_np,
                    "x_axis": x_axis_np,
                    "result_2d": result_2d,
                }
            except Exception as exc:
                st.error(f"The solver failed: {exc}")

last_result = st.session_state.get("last_solver_result")
if last_result is not None:
    st.divider()
    st.subheader("Results")
    st.caption(
        f"{last_result['problem_selection']} | "
        f"mu={last_result['mu']} | tau={last_result['tau']:g}"
    )
    current_inputs_match = (
        last_result["problem"] == problem
        and last_result["params"] == dict(params)
        and last_result["mu"] == int(num_eigen)
        and last_result["tau"] == float(tau)
    )
    if not current_inputs_match:
        st.info(
            "The plots below are the last completed run. One or more current solver "
            "inputs have changed; press Run solver to refresh these results."
        )

    result_problem = last_result["problem"]
    algorithm_np = last_result["algorithm_result"]
    actual_np = last_result["actual_result"]
    x_axis_np = last_result["x_axis"]

    if result_problem in {"OAF", "OAA"}:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis_np, actual_np, "b-", linewidth=3, label="PyTorch")
        ax.plot(x_axis_np, algorithm_np, "r.", markersize=10, label="TN")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.legend(loc="upper right")
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.download_button(
            label="Download figure",
            data=_figure_png(fig),
            file_name=f"{result_problem}_result.png",
            mime="image/png",
            key="download_oscillator_result",
            on_click="ignore",
        )
        plt.close(fig)

    elif result_problem == "C2D":
        comparison_col, heatmap_col = st.columns(2)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x_axis_np, actual_np, "b-", linewidth=2, label="PyTorch")
        ax1.plot(x_axis_np, algorithm_np, "r.", markersize=10, label="TN")
        ax1.set_xlabel("(x, y)")
        ax1.set_ylabel("T(x,y)")
        ax1.legend(loc="upper right")
        ax1.grid(False)
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        image = ax2.pcolormesh(last_result["result_2d"], cmap="CMRmap")
        fig2.colorbar(image, ax=ax2)
        ax2.set_xlabel("y")
        ax2.set_ylabel("x")
        ax2.grid(False)
        fig2.tight_layout()

        with comparison_col:
            st.pyplot(fig1)
            st.download_button(
                label="Download 1D comparison",
                data=_figure_png(fig1),
                file_name="C2D_1d_comparison.png",
                mime="image/png",
                key="download_c2d_1d",
                on_click="ignore",
            )

        with heatmap_col:
            st.pyplot(fig2)
            st.download_button(
                label="Download 2D heatmap",
                data=_figure_png(fig2),
                file_name="C2D_2d_heatmap.png",
                mime="image/png",
                key="download_c2d_2d",
                on_click="ignore",
            )

        plt.close(fig1)
        plt.close(fig2)

    mse = float(np.mean((algorithm_np - actual_np) ** 2))
    st.subheader("Error metric")
    st.metric("Mean Squared Error", f"{mse:.6f}")
