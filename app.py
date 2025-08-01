import streamlit as st
import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from solver import solve_problem

from matplotlib import rcParams
rcParams.update({
    'xtick.labelsize': 25,
    'ytick.labelsize': 25, 
    'font.size': 20,
    'axes.labelsize': 25,
    'legend.fontsize': 20
})

# Set page config
st.set_page_config(
    page_title="HHL Solver with Tensor Networks",
    page_icon="🧮",
    layout="wide"
)

# Create two columns for the entire layout
left_col, right_col = st.columns([1, 1])

with left_col:
    # Header
    st.title("🧮 HHL Algorithm Solver with Tensor Networks")
    st.markdown("""
    This application implements the HHL (Harrow-Hassidim-Lloyd) algorithm for solving systems of linear equations (Ax = b) 
    using tensor networks, based on the approach described in
    [Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective](https://arxiv.org/abs/2309.05290).
    
    Code developed by [Alejandro Mata Ali](https://github.com/DOKOS-TAYOS/Tensor_Networks_HHL_algorithm).
    
    The algorithm consists of:
    1. 📝 State preparation with input vector b
    2. 🔄 Quantum phase estimation to extract eigenvalues  
    3. 🎯 Controlled rotations based on eigenvalue estimates
    4. ↩️ Inverse phase estimation

    You can choose between three different problems to solve:
    - 🌊 **Forced Harmonic Oscillator**: Simulates a forced harmonic oscillator system
    - 🎵 **Damped Harmonic Oscillator**: Simulates a damped harmonic oscillator system  
    - 🌡️ **2D Heat Equation**: Solves the 2D heat equation
    """)

    # Problem selection
    problem_map = {
        "Forced Harmonic Oscillator": "OAF",
        "Damped Harmonic Oscillator": "OAA",
        "2D Heat Equation": "C2D"
    }
    problem_selection = st.selectbox(
        "🔍 Select the problem to solve",
        ["Forced Harmonic Oscillator", "Damped Harmonic Oscillator", "2D Heat Equation"],
        help="Choose which type of problem you want to solve"
    )
    problem = problem_map[problem_selection]

    # Parameter sliders for t and num_eigen
    col1, col2 = st.columns(2)
    with col1:
        default_t = 6000 if problem == 'OAF' else (11000 if problem == 'OAA' else 100)
        t = st.slider("⏱️ Time parameter (t)",
                      min_value=int(default_t/10),
                      max_value=int(default_t*10),
                      value=default_t,
                      help="Time parameter for the evolution operator")

    with col2:
        num_eigen = st.slider("🔢 Number of eigenvalues",
                             min_value=int(2000/10),
                             max_value=int(2000*10),
                             value=2000,
                             help="Number of eigenvalues to use in the calculation")

with right_col:
    # Default parameters for each problem
    default_params = {
        'OAF': {
            'k': 5.0,
            'm': 7.0,
            'nu': 3.14,
            'C': 9.0,
            'x0': 5.0,
            'xq': 3.0,
            'dt': 0.5,
            'steps': 100
        },
        'OAA': {
            'k': 5.0,
            'm': 7.0,
            'nu': 0.4,
            'C': 9.0,
            'x0': 5.0,
            'xq': 2.0,
            'dt': 0.5,
            'steps': 100,
            'gamma': 0.1
        },
        'C2D': {
            'k': 3.0,
            'u1x': 5.0,
            'u2x': 3.0,
            'u1y': 4.0,
            'u2y': 2.0,
            'dxy': 0.5,
            'nx': 20,
            'ny': 20
        }
    }

    # Create input fields for the selected problem's parameters
    st.subheader("⚙️ Problem Parameters")
    params = {}
    cols = st.columns(4)  # Create 4 columns for parameters
    param_list = list(default_params[problem].items())

    # Parameter descriptions for each problem type
    param_descriptions = {
        'OAF': {
            'k': '🔄 Spring constant (N/m)',
            'm': '⚖️ Mass (kg)',
            'nu': '📈 Natural frequency (rad/s)',
            'C': '💪 Driving force amplitude',
            'x0': '📍 Initial position (m)',
            'xq': '🏃 Initial velocity (m/s)',
            'dt': '⏲️ Time step (s)',
            'steps': '🔄 Number of simulation steps'
        },
        'OAA': {
            'k': '🔄 Spring constant (N/m)',
            'm': '⚖️ Mass (kg)',
            'nu': '📈 Natural frequency (rad/s)',
            'C': '💪 Driving force amplitude',
            'x0': '📍 Initial position (m)',
            'xq': '🏃 Initial velocity (m/s)',
            'dt': '⏲️ Time step (s)',
            'steps': '🔄 Number of simulation steps',
            'gamma': '📉 Damping coefficient'
        },
        'C2D': {
            'k': '🌡️ Thermal conductivity',
            'u1x': '🔥 Initial temperature at x boundary 1',
            'u2x': '🔥 Initial temperature at x boundary 2',
            'u1y': '🔥 Initial temperature at y boundary 1',
            'u2y': '🔥 Initial temperature at y boundary 2',
            'dxy': '📏 Grid spacing',
            'nx': '➡️ Number of x grid points',
            'ny': '⬆️ Number of y grid points'
        }
    }

    for i, (key, default_value) in enumerate(param_list):
        with cols[i % 4]:
            # Use integer input for steps, nx and ny parameters
            if key in ['steps', 'nx', 'ny']:
                params[key] = st.number_input(
                    param_descriptions[problem][key],
                    value=int(default_value),
                    step=1,
                    help=f"Parameter {key} for the {problem} problem"
                )
            else:
                params[key] = st.number_input(
                    param_descriptions[problem][key],
                    value=float(default_value),
                    step=0.1,
                    help=f"Parameter {key} for the {problem} problem"
                )

    # Run solver button
    if st.button("🚀 Run solver", type="primary"):
        with st.spinner("🔄 Solving the problem..."):
            try:
                # Run the solver
                algorithm_result, actual_result, x_axis, result_2d = solve_problem(
                    problem=problem,
                    params=params,
                    num_eigen=num_eigen,
                    t=t
                )

                # Create figures based on the problem type
                if problem in ['OAF', 'OAA']:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(x_axis, actual_result, 'b-', linewidth=3, label='PyTorch')
                    ax.plot(x_axis, algorithm_result, 'r.', markersize=10, label='TN')
                    ax.set_xlabel('t')
                    ax.set_ylabel('x')
                    ax.legend(loc='upper right')
                    ax.grid(False)
                    plt.tight_layout()

                    # Display the plot
                    st.subheader("📊 Results")
                    st.pyplot(fig)

                    # Save button
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    st.download_button(
                        label="💾 Download Figure",
                        data=buf.getvalue(),
                        file_name=f"{problem}_result.png",
                        mime="image/png"
                    )
                    plt.close(fig)  # Close the figure to free memory

                elif problem == 'C2D':
                    # 1D comparison plot
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(x_axis, actual_result, 'b-', linewidth=2, label='PyTorch')
                    ax1.plot(x_axis, algorithm_result, 'r.', markersize=10, label='TN')
                    ax1.set_xlabel('(x, y)')
                    ax1.set_ylabel('T(x,y)')
                    ax1.legend(loc='upper right')
                    ax1.grid(False)
                    plt.tight_layout()

                    # 2D heatmap
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    im = ax2.pcolormesh(result_2d, cmap="CMRmap")
                    plt.colorbar(im)
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                    ax2.grid(False)
                    plt.tight_layout()

                    # Display the plots
                    st.subheader("📊 Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig1)
                        buf1 = io.BytesIO()
                        fig1.savefig(buf1, format='png')
                        st.download_button(
                            label="💾 Download 1D Comparison",
                            data=buf1.getvalue(),
                            file_name=f"{problem}_1d_comparison.png",
                            mime="image/png"
                        )
                        plt.close(fig1)  # Close the figure to free memory

                    with col2:
                        st.pyplot(fig2)
                        buf2 = io.BytesIO()
                        fig2.savefig(buf2, format='png')
                        st.download_button(
                            label="💾 Download 2D Heatmap",
                            data=buf2.getvalue(),
                            file_name=f"{problem}_2d_heatmap.png",
                            mime="image/png"
                        )
                        plt.close(fig2)  # Close the figure to free memory

                # Display error metrics
                st.subheader("📉 Error Metrics")
                # Convert tensors to numpy arrays and calculate MSE
                alg_np = algorithm_result.detach().numpy() if isinstance(algorithm_result, torch.Tensor) else algorithm_result
                act_np = actual_result.detach().numpy() if isinstance(actual_result, torch.Tensor) else actual_result
                error = np.mean((alg_np - act_np) ** 2)
                st.metric("Mean Squared Error", f"{error:.6f}")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")