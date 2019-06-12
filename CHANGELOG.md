# Changelog

All notable changes to Kite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Fixed
- ISCE import: XML tag cases

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
