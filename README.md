# CRoadA

## Project Overview

**CRoadA** is a project focused on augmenting datasets of road grids by generating new, realistic examples. Using deep learning, the system learns to create synthetic road grid samples based on provided city grid data. This approach enables the expansion of existing datasets with diverse examples, supporting research and development in areas such as urban planning, autonomous driving, and geographic data analysis. The grid includes parameters such as altitude, street presence, and whether a street is residential.

## Features

- Batch generation from large grid files
- Grid segment reading and writing utilities
- Extensible for various grid types and training scenarios

## Directory Structure

```
CRoadA/
├── application
│   ├── __init__.py
│   ├── main.py
│   ├── main_window.py
│   ├── MapWindow.py
│   ├── static
│   │   ├── main.css
│   │   └── map.js
│   ├── templates
│   │   ├── map.html
│   │   └── map-style.css
│   └── ui_manager.py
├── cache
├── check_dir
│   └── check.grid
├── curve_analizer
│   └── curve_analizer.py
├── debug_segments
├── .gitignore
├── graph_remaker
│   ├── borders_identifier.py
│   ├── data_analyser.py
│   ├── data_structures.py
│   ├── __init__.py
│   ├── integrated.py
│   ├── memory_wise.py
│   ├── morphological_remaker.py
│   ├── prediction_statistics.py
│   ├── __pycache__
│   ├── streets_separator.py
│   └── utils.py
├── grid_manager.py
├── grids                          -> city road grid files to train models
│   ├── evaluation
│   ├── square_test.dat
│   └── with-is-residential
│       ├── Bydgoszcz-Polska.city_grid
│       ├── Chorzów-Polska.city_grid
│       ├── Gdańsk-Polska.city_grid
│       ├── Gdynia-Polska.city_grid
│       ├── Koszalin-Polska.city_grid
│       ├── Lublin-Polska.city_grid
│       ├── Mielec-Polska.city_grid
│       ├── Radom-Polska.city_grid
│       ├── Rybnik-Polska.city_grid
│       ├── Świętochłowice-Polska.city_grid
│       └── Toruń-Polska.city_grid
├── __init__.py
├── main.ipynb                    -> an overview for visualisation of the whole project
├── models
│   └── shallowed_unet_256_1m_is_street_only      -> sample model save
│       ├── 1770922591_model.keras
│       └── 1770926494_model.keras
├── __pycache__
├── README.md                     -> this README
├── requirements-no-gpu.txt
├── requirements.txt
├── resegment.ipynb
├── scraper
│   ├── data_loader.py
│   ├── geometry_processor.py
│   ├── graph_loader.py
│   ├── grid_builder.py
│   ├── grid.ipynb
│   ├── __init__.py
│   ├── locator.py
│   ├── __pycache__
│   ├── rasterizer.py
│   ├── requirements.txt
│   └── scrapper_miast.ipynb
├── test_2.ipynb
├── test_check_dir
│   └── check.grid
├── test_images
├── testing_field.py
├── test_m.py
├── test.py
├── tmp
└── trainer
    ├── clipping_model.py         -> model main logic
    ├── cut_grid.py               -> cut from/to files utils
    ├── data_generator.py         -> cut generation logic (for batches in keras model)
    ├── model_architectures.py    -> place for models used in clipping_model.py
    ├── model_metrics.py
    ├── model.py                  -> abstract class for the model
    ├── __pycache__
    └── trainer.py                -> "main" class for the trainer module (interface for training)
```

## Requirements

- Python 3.12
  [file with requirements](./requirements.txt)
  [requirements-no-gpu](./requirements-no-gpu.txt)

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

## Authors

- CRoadA team ([CRoadA](https://github.com/CRoadA))

---

_For questions or contributions, please open an issue or submit a pull request!_
