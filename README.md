# Classic Machine Learning TUI

A modern, terminal-based interactive suite for exploring fundamental Machine Learning algorithms. Built with **Textual** and **Plotext** for a seamless, live-updating experience.

## Features

- **Live Calculation**: Adjust parameters (K, Learning Rate, Epochs) and see results update instantly.
- **Interactive Visualizations**: High-resolution terminal charts for K-Means, Linear Regression, and Perceptron.
- **Multilingual**: Full support for English and Spanish.
- **Split-View Design**: Inputs on the left, visual data on the right.

## Algorithms Included

1.  **Perceptron**: Binary classifier with live decision boundary visualization.
2.  **Linear Regression**: Predicting continuous values with optional Ridge Regularization (λ).
3.  **K-Means**: Unsupervised clustering with live centroid updates.

## Quick Start

You can now install `clml` directly via pip and run it using the dedicated command.

### Installation

```bash
pip install clml
```

### Running the App

Simply type the following in your terminal:

```bash
clml
```

## Development & Local Setup

If you want to run the project from source:

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -e .
    ```
3.  Run the app:
    ```bash
    python -m classic_ml.app
    ```

## Controls

- `Q`: Quit the application.
- `L`: Toggle between English and Spanish.
- `I`: Load the built-in **Iris** dataset for immediate testing.
- `R`: Reset all parameters to default values.

## License

This project is distributed under the MIT license.
