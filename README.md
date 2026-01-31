# Partially Distinguishable Boson Sampling

This project implements a solver for calculating the probability of detecting specific particle distributions at the output of an interferometer, given partially distinguishable particles.

Based on the paper: *Sampling of partially distinguishable bosons and the relation to the multidimensional permanent* (arXiv:1410.7687v3).

## Structure

*   `src/solver.py`: Contains the `calculate_probability` function implementing the Tensor Permanent algorithm (Generalized Ryser).
*   `tests/test_simulation.py`: Unit tests verifying the Hong-Ou-Mandel effect and other limits.
*   `run_demo.py`: A simple demonstration script.

## Usage

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the demo:
    ```bash
    python run_demo.py
    ```

3.  Run tests:
    ```bash
    pytest tests/test_simulation.py
    ```
