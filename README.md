# GSlibJax
![GSlibJax Logo](gslib_jax.png)
## A JAX-powered Reimagining of GSlib

GSlibJax is a high-performance, JAX-native implementation of the Geostatistical Software Library (GSlib). Leveraging JAX's capabilities for automatic differentiation, GPU acceleration, and XLA compilation, GSlibJax aims to provide a modern, flexible, and efficient toolkit for geostatistical modeling and simulation.

## Why GSlibJax?

GSlib has long been the gold standard for geostatistical applications. However, in the era of deep learning and large-scale data analysis, there's a growing need for geostatistical tools that can seamlessly integrate with modern scientific computing workflows. GSlibJax addresses this by:

* **JAX Native:** Built from the ground up in JAX, providing automatic differentiation for gradient-based optimization, just-in-time (JIT) compilation, and execution on CPUs, GPUs, and TPUs.
* **Performance:** Achieve significant speedups on complex geostatistical tasks, especially with large datasets and intensive simulations, by utilizing JAX's XLA compilation.
* **Flexibility:** Easily integrate geostatistical models into larger machine learning pipelines or custom scientific applications.
* **Modern API:** A clean, Pythonic API designed for ease of use and readability, while maintaining the core functionalities of GSlib.
* **Reproducibility:** JAX's functional programming paradigm encourages more reproducible and testable code.

## Features (Planned)

GSlibJax is under active development, with the following core functionalities planned:

* **Variogram Modeling:**
    * Experimental variogram calculation
    * Theoretical variogram fitting (Spherical, Exponential, Gaussian, Power, etc.)
* **Kriging:**
    * Simple Kriging
    * Ordinary Kriging
    * Universal Kriging
    * Indicator Kriging
    * CoKriging
* **Simulation:**
    * Sequential Gaussian Simulation (SGS)
    * Sequential Indicator Simulation (SIS)
* **Data Handling:**
    * Gridding and interpolation utilities
    * Data conditioning
* **Utilities:**
    * Neighborhood search algorithms
    * Matrix solvers optimized for geostatistical problems

## Installation

GSlibJax can be installed via pip:

```bash
pip install gslibjax
