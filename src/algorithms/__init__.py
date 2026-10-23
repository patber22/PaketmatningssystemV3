"""
Algoritmlager för Paketmätningssystem V3.

Innehåller moduler för 3D-djupberäkning (StereoDepthEstimator),
ytnormaler (PhotometricStereoEstimator), bordsmodellering (BedModel)
och sensorfusion (DimensionFuser).
"""

from src.algorithms.stereo_depth import StereoDepthEstimator

__all__ = ["StereoDepthEstimator"]
