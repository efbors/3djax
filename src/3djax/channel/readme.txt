channel sub-project layout

src/3djax/
├── channel/                    # The Environment
│   ├── docs/                   # Step 1:  (Slides, diagrams, PDFs)
│   ├── policies/               # Step 2: Policy Configuration
│   │   ├── base_config.yaml    # Shared baseline parameters
│   │   ├── class_a_bringup.yaml
│   │   └── class_c_dynamic.yaml
│   ├── models/                 # The Mathematical Implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract Base Class (The API Contract)
│   │   ├── physical.py         # E.g. the double-exponential analytical model
│   │   └── sparameter.py       # Future S-parameter Touchstone ingestion
│   ├── orchestrator.py         # Step 3: NumPy Trajectory Generator (Pass 1 & 2)
│   └── runtime.py              # Step 4: JAX Block Engine (Overlap-save, cross-fade)
|
|... rest of the project