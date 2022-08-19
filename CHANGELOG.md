# Changelog

All notable changes to Kite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.0] 27. April 2020

### Added
- Import of SNAP products
- Scene: Added method `get_elevation`
- Scene: Added polygon masking of unwanted areas
- Scene: Added empirical APS removal
- Scene: Added GACOS APS import and removal
- Scene: Added Deramping as module
- SceneStack: handles stacks of time series
- Spool: Added Plugin handlers

### Changed
- QuadTree: Exporting unit vectors [ENU]

### Fixed
- Spool: improved UI
- Covariance: fixing sparse spool noise window selection
- Covariance: fixed spectral 1D estmiation // performance improvements

## [1.3.0] 3. December 2019

### Added
- Spool: Added outline to Quadtree Nodes
- Spool: Added scene de-ramp action
- `bbd2kite`: import of BGR BodenBewegungsDienst data (https://bodenbewegungsdienst.bgr.de)

### Changed
- Spool:
  - changed leaf center color
  - changed satellite arrow label
  - Quadtree sliders now behave exponential for better fine-tuning
  - Performance upgrade
- `stamps2kite` parses pixel size in meters instead of grid size
- Quadtree smallest node size is calculated adaptively

### Fixed
- Spool:
  - Fixed displayed coordinates
  - Improvement slider responsiveness
- Gamma: Fixed ncols/nrows import
- Quadtree: focal point calculation to accommodate small quadtree nodes.
- Quadtree: Export UTM GeoJSON

## [1.2.4] 29. October 2019

### Added
- Added ARIA import
- `quadtree` added GeoJSON export
- Spool: covariance progress bar

### Fixed
- Spool: added absolute degree to cursor position
- bugfixes LiCSAR import

## [1.2.3] 17. July 2019

### Added
- `Scene` added `__i/add__`, `__i/sub__` and `__neg__` standard operators.
- Talpa dialog to change LOS vectors (phi and theta) of forward modelled scene.

### Fixed
- MacOS gcc install with OpenMP.
- `stamps2kite` Backward compat MATLAB import.
- `covariance`/`quadtree` more defensive against sparsely correlated scenes.
- LiCSAR import more defensive data handling.

## [1.2.2] 26. June 2019

### Added
- scene displacement pixel variance attribute`displacement_px_var`.
- quadtree and covariance propagation of `displacement_px_var` towards diagonal.
- spool to display `displacement_px_var`.
- More meta information for `stamps2kite`.

### Fixed
- `stamps2kite`: look angle import

## [1.2.1] 17. June 2019

### Fixed
- `stamps2kite` import of look angle is now from interpolation.

## [1.2.0] 14. June 2019

### Added
- `stamps2kite` conversion tool.

### Fixed
- ISCE import: XML tag cases.
- Version numbering in `setup.py`

## [1.1.1] 6. June 2019

### Added
- Deramping of displacement maps `Scene.displacement_deramp()`

### Fixed
- Spool more digits for degree scenes
- Fixed log handling and handler injection

## [1.1.0] 6. May 2019

### Added
- Added LiCSAR import
- Added SARSCAPE import
- Added LiCSAR downloader (`client.py`)
- Added CLVDVolume to talpa
- Added SpinBox for sliders

### Fixed
- talpa various improvements and bugfixes
- updated documentation
- fixed import for ROIPAC and ISCE
- matplotlib plotting functions

### Changed
- Spool: Changes 'LOS' to 'Towards Satellite'

## [1.0.1] 21. January 2019

### Added
- Adapting semating versioning
- Spool: added log control

### Fixed
- Improved log control
- Documentation fixes and internal references.
- Fixed Gamma import from Lat/Lon
- docs: Removed Anaconda pre-build installation
