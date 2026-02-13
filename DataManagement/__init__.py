"""
DataManagement - A flexible data container with query capabilities.

This package provides DataPoint and DataSerials classes for managing
collections of data with powerful filtering, aggregation, and serialization.
"""

__version__ = '0.1.0'
__author__ = 'Shihong Chen'
__email__ = 'schenek@connect.ust.hk'

from .core import DataPoint, DataSerials

__all__ = ['DataPoint', 'DataSerials']
