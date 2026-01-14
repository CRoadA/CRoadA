# CRoadA

## Project Overview

**CRoadA** is a project focused on augmenting datasets of road grids by generating new, realistic examples. Using deep learning, the system learns to create synthetic road grid samples based on provided city grid data. This approach enables the expansion of existing datasets with diverse examples, supporting research and development in areas such as urban planning, autonomous driving, and geographic data analysis. The grid includes parameters such as altitude, street presence, and whether a street is residential.

## Features

- Batch generation from large grid files
- Randomized and deterministic cut sequences for data augmentation
- Support for multi-threaded data loading via Keras Sequences
- Grid segment reading and writing utilities
- Extensible for various grid types and training scenarios

## Directory Structure

```
CRoadA/
├── application
│   ├── main.py
│   ├── test_MapDisplayer.py
│   └── ui_manager.py
├── cache
│   ├── c737f046323216334159224013f1a92b4c71a9dc.json
│   └── ...
├── check_dir
│   └── check.grid
├── curve_analizer
│   └── curve_analizer.py
├── .gitignore
├── graph_remaker
│   ├── borders_identifier.py
│   ├── data_analyser.py
│   ├── data_structures.py
│   ├── integrated.py
│   ├── memory_wise.py
│   ├── morphological_remaker.py
│   ├── prediction_statistics.py
│   ├── __pycache__
│   │   └── ...
│   ├── streets_separator.py
│   └── utils.py
├── grid_manager.py
├── grids
│   ├── Gliwice.city_grid
│   ├── przyklad1.dat
│   ├── square_test.dat
│   ├── Tychy.dat
│   ├── Zabrze.city_grid
│   └── ...
├── __init__.py
├── main.ipynb
├── __pycache__
│   └── ...
├── README.md
├── requirements.txt
├── scraper
│   ├── cache
│   │   ├── fe8eb1852dff9af8c4881e473737b87098263a17.json
│   │   └── ...
│   ├── data_loader.py
│   ├── geometry_processor.py
│   ├── graph_loader.py
│   ├── grid_builder.py
│   ├── grid.ipynb
│   ├── grids
│   ├── __init__.py
│   ├── __pycache__
│   │   └── ...
│   ├── rasterizer.py
│   ├── requirements.txt
│   └── scrapper_miast.ipynb
├── test_2.ipynb
├── test_check_dir
│   └── check.grid
├── test_images
│   ├── crossroad_conflict.png
│   └── ...
├── testing_field.py
├── test_m.py
├── test.py
├── tmp
│   └── batches
│       └── batch_sequences
│           └── cuts
│               ├── przyklad1.dat_cut_1034_28_1000_1000.dat
│               └── ...
└── trainer
    ├── batch_sequence.py
    ├── clipping_model.py
    ├── clipping_sequence.py
    ├── model.py
    ├── __pycache__
    │   └── ...
    └── trainer.py
```

## Requirements

- Python 3.12
  [file with requirements](./requirements.txt)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CRoadA/CRoadA.git
   cd CRoadA
   ```

2. **Check [main.ipynb notebook file](./main.ipynb)** for the tutorial.

## Example

### TODO

## Authors

- CRoadA team ([CRoadA](https://github.com/CRoadA))

---

_For questions or contributions, please open an issue or submit a pull request!_
