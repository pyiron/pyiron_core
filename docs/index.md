# Overview

`pyiron_core` is a graph‑based workflow engine for materials‑science simulations. It combines a programmatic API with a visual scripting GUI, supporting Jupyter notebooks, hash‑based result caching, and a pre‑built library of domain‑specific nodes.

## Key Features
- **Hybrid workflow creation** – mix code‑based and visual‑scripting approaches.
- **Jupyter‑native** – create, inspect, and run nodes directly in notebooks.
- **Boilerplate‑free** – turn a regular Python function into a graph node with a single decorator.
- **Hash‑based caching** – identical sub‑graphs reuse cached results for fast iteration.
- **Inclusive node library** – ready‑made nodes for atomistics, electrochemistry, databases, and more.

## Short video tutorial
<video width="100%" controls poster="./pyiron_core_demo_first_frame.jpg">
  <source src="./pyiron_core_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Quick Demo
![DEMO](./pyiron_core_demo_snippet.gif)

The documentation below expands on installation, usage, and development.
