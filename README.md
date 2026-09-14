# Real-Time WebAR Scene Understanding and Topological Parsing Engine

An optimized, client-side computer vision application designed for asynchronous multi-object spatial analysis via hybrid runtime pipelines.

## Core Features
*   **Throttled Inference Cadence:** Implements a strict 2000ms scan execution loop to optimize hardware performance and prevent thermal throttling on mobile devices.
*   **Target Domain Constraints:** Hard-capped non-maximum suppression sorting matrix parameters to a maximum of 2 targets to minimize thread choke.
*   **Topological Spatial Parsing Engine:** Custom geometric midpoint algorithms calculating real-time relative boundaries ("on top of", "below", "beside").
*   **Local-Only Processing:** Zero backend dependency or cloud API queries; leverages client-side WebGL acceleration for total data privacy.

## Tech Stack
*   **Framework:** TanStack Start (React + TypeScript)
*   **Styling:** Tailwind CSS (Night Instrument Design System)
*   **Machine Learning Ecosystem:** TensorFlow.js with the COCO-SSD object detection topology
*   
