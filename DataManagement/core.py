"""
DataPoint and DataSerials - A flexible data container with query capabilities
"""

import copy
import sys
import re
import warnings
import json
import pickle
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union, Callable, Iterator, Set
from collections import defaultdict
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class DataPoint:
    """
    Flexible data container with dynamic attributes and serialization support.
    """
    
    __slots__ = ('_data', '_id')  # Memory optimization
    
    def __init__(self, id: Optional[str] = None, **kwargs):
        """
        Initialize DataPoint with optional id and attributes.
        
        Args:
            id: Unique identifier for the DataPoint
            **kwargs: Additional attributes
        """
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_id', None)
        
        if id is not None:
            self._id = id
        
        for key, value in kwargs.items():
            self[key] = value
    
    @property
    def id(self) -> Optional[str]:
        """Get the DataPoint's id"""
        return self._id
    
    @id.setter
    def id(self, value: str) -> None:
        """Set the DataPoint's id"""
        if not isinstance(value, str):
            raise TypeError(f"id must be string, got {type(value).__name__}")
        object.__setattr__(self, '_id', value)
    
    def __getitem__(self, key: str) -> Any:
        """Get attribute using dictionary syntax"""
        return self._data.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set attribute using dictionary syntax"""
        if key.startswith('_'):
            raise ValueError(f"Attribute name cannot start with underscore: '{key}'")
        self._data[key] = value
    
    def __delitem__(self, key: str) -> None:
        """Delete attribute"""
        del self._data[key]
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation"""
        if name.startswith('_') or name not in self._data:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        return self._data[name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
    
    def __delattr__(self, name: str) -> None:
        """Delete attribute using dot notation"""
        if name in self._data:
            del self._data[name]
        else:
            object.__delattr__(self, name)
    
    def __contains__(self, key: str) -> bool:
        """Check if attribute exists"""
        return key in self._data
    
    def __len__(self) -> int:
        """Number of attributes (excluding id)"""
        return len(self._data)
    
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over attributes"""
        yield 'id', self._id
        yield from self._data.items()
    
    def __eq__(self, other: object) -> bool:
        """Compare DataPoints by id"""
        if not isinstance(other, DataPoint):
            return NotImplemented
        return self._id == other._id and self._data == other._data
    
    def __hash__(self) -> int:
        """Hash based on id and frozen data"""
        return hash((self._id, frozenset(self._data.items())))
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        parts = []
        if self._id is not None:
            parts.append(f"id='{self._id}'")
        for key, value in self._data.items():
            parts.append(f"{key}={repr(value)}")
        return f"DataPoint({', '.join(parts)})"
    
    def __str__(self) -> str:
        """Readable string representation"""
        return self.__repr__()
    
    def __copy__(self) -> 'DataPoint':
        """Create a shallow copy"""
        new_dp = DataPoint(id=self._id)
        new_dp._data = self._data.copy()
        return new_dp
    
    def __deepcopy__(self, memo: dict) -> 'DataPoint':
        """Create a deep copy"""
        new_dp = DataPoint(id=copy.deepcopy(self._id, memo))
        new_dp._data = copy.deepcopy(self._data, memo)
        memo[id(self)] = new_dp
        return new_dp
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default value"""
        return self._data.get(key, default)
    
    def keys(self) -> Set[str]:
        """Get all attribute keys"""
        return set(self._data.keys())
    
    def values(self) -> List[Any]:
        """Get all attribute values"""
        return list(self._data.values())
    
    def items(self) -> List[Tuple[str, Any]]:
        """Get all attribute items"""
        return list(self._data.items())
    
    def update(self, **kwargs) -> 'DataPoint':
        """Update multiple attributes"""
        for key, value in kwargs.items():
            self[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = self._data.copy()
        if self._id is not None:
            result['id'] = self._id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPoint':
        """Create DataPoint from dictionary"""
        data = data.copy()
        id = data.pop('id', None)
        return cls(id=id, **data)
    
    def to_xml(self) -> ET.Element:
        """Convert to XML element"""
        elem = ET.Element('DataPoint')
        if self._id is not None:
            elem.set('id', self._id)
        
        for key, value in self._data.items():
            attr_elem = ET.SubElement(elem, 'attr', name=key)
            self._value_to_xml(attr_elem, value)
        
        return elem
    
    @staticmethod
    def _value_to_xml(parent: ET.Element, value: Any) -> None:
        """Convert value to XML representation"""
        if value is None:
            parent.set('type', 'none')
        elif isinstance(value, bool):
            parent.set('type', 'bool')
            parent.set('value', str(value).lower())
        elif isinstance(value, int):
            parent.set('type', 'int')
            parent.set('value', str(value))
        elif isinstance(value, float):
            parent.set('type', 'float')
            parent.set('value', str(value))
        elif isinstance(value, str):
            parent.set('type', 'str')
            parent.text = value
        elif isinstance(value, (list, tuple)):
            parent.set('type', 'list')
            for item in value:
                item_elem = ET.SubElement(parent, 'item')
                parent.text = str(item) if item is not None else ''
        elif isinstance(value, dict):
            parent.set('type', 'dict')
            for k, v in value.items():
                item_elem = ET.SubElement(parent, 'item', key=str(k))
                item_elem.text = str(v) if v is not None else ''
        elif isinstance(value, datetime):
            parent.set('type', 'datetime')
            parent.set('value', value.isoformat())
        else:
            parent.set('type', 'str')
            parent.text = str(value)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'DataPoint':
        """Create DataPoint from XML element"""
        id = elem.get('id')
        dp = cls(id=id)
        
        for attr_elem in elem.findall('attr'):
            name = attr_elem.get('name')
            if not name:
                continue
            
            value = cls._value_from_xml(attr_elem)
            dp[name] = value
        
        return dp
    
    @staticmethod
    def _value_from_xml(elem: ET.Element) -> Any:
        """Extract value from XML element"""
        type_ = elem.get('type', 'str')
        
        if type_ == 'none':
            return None
        elif type_ == 'bool':
            return elem.get('value', 'false').lower() == 'true'
        elif type_ == 'int':
            return int(elem.get('value', '0'))
        elif type_ == 'float':
            return float(elem.get('value', '0.0'))
        elif type_ == 'datetime':
            return datetime.fromisoformat(elem.get('value', ''))
        elif type_ == 'str':
            return elem.text or ''
        elif type_ == 'list':
            return [item.text for item in elem.findall('item')]
        elif type_ == 'dict':
            return {item.get('key', ''): item.text or '' 
                   for item in elem.findall('item')}
        else:
            return elem.text or ''


class DataSerials(MutableSequence):
    """
    A powerful container for DataPoint objects with query, filter, and serialization capabilities.
    
    Features:
    - List-like access with slicing
    - Dict-like access by id
    - Powerful filtering with Django-like syntax
    - Aggregation and annotation
    - XML/JSON serialization
    - Method chaining
    """
    
    def __init__(self, data: Optional[List[DataPoint]] = None):
        """
        Initialize DataSerials.
        
        Args:
            data: Optional initial list of DataPoint objects
        """
        self._data: List[DataPoint] = []
        self._dict: Dict[str, int] = {}  # id -> index mapping
        self._metadata: Dict[str, Any] = {}
        
        if data:
            for dp in data:
                self.append(dp)
    
    # ============ Core Data Management ============
    
    def append(self, dp: DataPoint, force_update: bool = False) -> None:
        """
        Add a DataPoint to the collection.
        
        Args:
            dp: DataPoint to add
            force_update: If True, overwrite existing DataPoint with same id
        
        Raises:
            ValueError: If DataPoint has no id and force_update is False
        """
        if dp.id is None:
            raise ValueError('DataPoint must have an id attribute')
        
        if dp.id in self._dict:
            if not force_update:
                raise KeyError(
                    f"DataPoint with id '{dp.id}' already exists. "
                    "Use force_update=True to overwrite."
                )
            # Update existing
            idx = self._dict[dp.id]
            self._data[idx] = dp
        else:
            # Add new
            idx = len(self._data)
            self._data.append(dp)
            self._dict[dp.id] = idx
    
    def insert(self, index: int, dp: DataPoint) -> None:
        """Insert DataPoint at specified index"""
        if dp.id is None:
            raise ValueError('DataPoint must have an id attribute')
        
        if dp.id in self._dict:
            raise KeyError(f"DataPoint with id '{dp.id}' already exists")
        
        self._data.insert(index, dp)
        # Rebuild index mapping
        self._rebuild_index()
    
    def extend(self, dps: List[DataPoint]) -> None:
        """Extend collection with multiple DataPoints"""
        for dp in dps:
            self.append(dp)
    
    def remove(self, dp: Union[DataPoint, str]) -> None:
        """
        Remove a DataPoint by id or reference.
        
        Args:
            dp: DataPoint object or id string
        """
        if isinstance(dp, DataPoint):
            if dp.id is None:
                raise ValueError('Cannot remove DataPoint without id')
            dp_id = dp.id
        else:
            dp_id = dp
        
        if dp_id not in self._dict:
            raise KeyError(f"DataPoint with id '{dp_id}' not found")
        
        idx = self._dict[dp_id]
        del self._data[idx]
        del self._dict[dp_id]
        
        # Update indices for remaining items
        self._rebuild_index()
    
    def pop(self, index: int = -1) -> DataPoint:
        """Remove and return DataPoint at index"""
        dp = self._data.pop(index)
        del self._dict[dp.id]
        self._rebuild_index()
        return dp
    
    def clear(self) -> None:
        """Remove all DataPoints"""
        self._data.clear()
        self._dict.clear()
        self._metadata.clear()
    
    def _rebuild_index(self) -> None:
        """Rebuild the id->index mapping"""
        self._dict = {dp.id: i for i, dp in enumerate(self._data) if dp.id is not None}
    
    # ============ Metadata Management ============
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary (read-only view)"""
        return self._metadata.copy()
    
    def set_metadata(self, key: str, value: Any) -> 'DataSerials':
        """Set metadata value"""
        if key.startswith('_'):
            raise ValueError(f"Metadata key cannot start with underscore: '{key}'")
        self._metadata[key] = value
        return self
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self._metadata.get(key, default)
    
    def update_metadata(self, **kwargs) -> 'DataSerials':
        """Update multiple metadata values"""
        for key, value in kwargs.items():
            self.set_metadata(key, value)
        return self
    
    # ============ Access Methods ============
    
    def __getitem__(self, key: Union[int, str, slice]) -> Union[DataPoint, 'DataSerials']:
        """
        Access DataPoints by:
        - Integer index -> returns DataPoint
        - String id -> returns DataPoint
        - Slice -> returns new DataSerials instance
        """
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, str):
            if key not in self._dict:
                raise KeyError(f"DataPoint with id '{key}' not found")
            return self._data[self._dict[key]]
        elif isinstance(key, slice):
            new_ds = self.__class__()
            new_ds._metadata = copy.deepcopy(self._metadata)
            new_ds._data = [copy.deepcopy(dp) for dp in self._data[key]]
            new_ds._rebuild_index()
            return new_ds
        else:
            raise TypeError(f'Invalid key type: {type(key).__name__}')
    
    def __setitem__(self, key: Union[int, str], dp: DataPoint) -> None:
        """
        Set DataPoint at index or by id.
        Operation is atomic - state is unchanged if any error occurs.
        """

        # Handle integer key
        if isinstance(key, int):
            # Normalize negative index
            idx = key if key >= 0 else len(self._data) + key

            # Check bounds
            if idx < 0 or idx >= len(self._data):
                raise IndexError(f'Index {key} out of range for DataSerials with size {len(self)}')

            old_dp = self._data[idx]

            # Check for conflicts BEFORE making any changes
            if dp.id is not None and dp.id in self._dict and self._dict[dp.id] != idx:
                raise KeyError(f"DataPoint with id '{dp.id}' already exists at position {self._dict[dp.id]}")

            # All checks passed - now perform the update atomically
            if old_dp.id in self._dict:
                del self._dict[old_dp.id]

            self._data[idx] = dp

            if dp.id is not None:
                self._dict[dp.id] = idx

        # Handle string key
        elif isinstance(key, str):
            if key not in self._dict:
                raise KeyError(f"DataPoint with id '{key}' not found")

            idx = self._dict[key]
            old_dp = self._data[idx]

            # Check if new DataPoint has id
            if dp.id is None:
                raise ValueError(f"Cannot assign DataPoint without id to key '{key}'")

            # Check for conflicts BEFORE making any changes
            if dp.id != key and dp.id in self._dict:
                raise KeyError(f"DataPoint with id '{dp.id}' already exists at position {self._dict[dp.id]}")

            # All checks passed - now perform the update atomically
            if old_dp.id in self._dict:
                del self._dict[old_dp.id]

            self._data[idx] = dp
            self._dict[dp.id] = idx

        else:
            raise TypeError(f'Invalid key type: {type(key).__name__}')
    
    def __delitem__(self, key: Union[int, str]) -> None:
        """Delete DataPoint by index or id"""
        if isinstance(key, int):
            dp = self._data.pop(key)
            if dp.id in self._dict:
                del self._dict[dp.id]
            self._rebuild_index()
        elif isinstance(key, str):
            self.remove(key)
        else:
            raise TypeError(f'Invalid key type: {type(key).__name__}')
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self._data)
    
    def __contains__(self, key: Union[str, DataPoint]) -> bool:
        """Check if id or DataPoint exists"""
        if isinstance(key, DataPoint):
            return key.id in self._dict if key.id is not None else False
        return key in self._dict
    
    def __reversed__(self) -> Iterator[DataPoint]:
        return reversed(self._data)
    
    # ============ Query & Filter API ============
    
    def filter(self, *conditions: Callable[[DataPoint], bool], **kwargs) -> 'DataSerials':
        """
        Filter DataPoints with powerful query capabilities.
        
        Args:
            *conditions: Variable number of predicate functions
            **kwargs: Field-based conditions with operators (field__op=value)
            
        Returns:
            New DataSerials instance with filtered DataPoints
            
        Examples:
            ds.filter(age__gte=18)
            ds.filter(name__contains='John')
            ds.filter(lambda dp: dp.score > 85)
            ds.filter(age__gte=18, age__lte=65)
            ds.filter(status__in=['active', 'pending'])
        """
        result = self.__class__()
        result._metadata = copy.deepcopy(self._metadata)
        
        for dp in self._data:
            if self._matches(dp, conditions, kwargs):
                result.append(copy.deepcopy(dp), force_update=True)
        
        return result
    
    def exclude(self, *conditions: Callable[[DataPoint], bool], **kwargs) -> 'DataSerials':
        """
        Exclude DataPoints that match conditions.
        
        Examples:
            ds.exclude(status='inactive')
            ds.exclude(age__lt=18)
        """
        result = self.__class__()
        result._metadata = copy.deepcopy(self._metadata)
        
        for dp in self._data:
            if not self._matches(dp, conditions, kwargs):
                result.append(copy.deepcopy(dp), force_update=True)
        
        return result
    
    def get(self, **kwargs) -> Optional[DataPoint]:
        """Get a single DataPoint matching conditions."""
        filtered = self.filter(**kwargs)
        if len(filtered) > 1:
            raise ValueError(f"get() returned {len(filtered)} DataPoints, expected 1")
        return filtered[0] if filtered else None
    
    def get_or_none(self, **kwargs) -> Optional[DataPoint]:
        """Get a single DataPoint or None if not found/exceptions."""
        try:
            return self.get(**kwargs)
        except (KeyError, ValueError):
            return None
    
    def first(self) -> Optional[DataPoint]:
        """Get first DataPoint or None if empty."""
        return self._data[0] if self._data else None
    
    def last(self) -> Optional[DataPoint]:
        """Get last DataPoint or None if empty."""
        return self._data[-1] if self._data else None
    
    @staticmethod
    def _matches(dp: DataPoint, conditions: Tuple[Callable, ...], 
                field_conditions: Dict[str, Any]) -> bool:
        """Check if DataPoint matches all conditions."""
        # Custom function conditions
        for condition in conditions:
            try:
                if not condition(dp):
                    return False
            except Exception:
                return False
        
        # Field conditions
        for key, value in field_conditions.items():
            if not DataSerials._evaluate(dp, key, value):
                return False
        
        return True
    
    @staticmethod
    def _evaluate(dp: DataPoint, key: str, value: Any) -> bool:
        """Evaluate a single field condition."""
        # Parse operator
        if '__' in key:
            field, op = key.rsplit('__', 1)
        else:
            field, op = key, 'eq'
        
        # Get value
        dp_value = dp.get(field)
        
        # Handle operators
        if op == 'eq':
            return dp_value == value
        elif op == 'ne':
            return dp_value != value
        elif op == 'gt':
            return dp_value > value
        elif op == 'gte':
            return dp_value >= value
        elif op == 'lt':
            return dp_value < value
        elif op == 'lte':
            return dp_value <= value
        elif op == 'in':
            return dp_value in value
        elif op == 'contains':
            return value.lower() in str(dp_value).lower()
        elif op == 'startswith':
            return str(dp_value).startswith(str(value))
        elif op == 'endswith':
            return str(dp_value).endswith(str(value))
        elif op == 'isnull':
            return dp_value is None if value else dp_value is not None
        elif op == 'between':
            return value[0] <= dp_value <= value[1] if len(value) == 2 else False
        elif op == 'regex':
            return bool(re.search(value, str(dp_value)))
        elif op == 'len':
            return len(dp_value) == value
        elif op == 'len_gt':
            return len(dp_value) > value
        elif op == 'len_lt':
            return len(dp_value) < value
        elif op == 'year' and hasattr(dp_value, 'year'):
            return dp_value.year == value
        elif op == 'month' and hasattr(dp_value, 'month'):
            return dp_value.month == value
        elif op == 'day' and hasattr(dp_value, 'day'):
            return dp_value.day == value
        elif op == 'weekday' and hasattr(dp_value, 'weekday'):
            return dp_value.weekday() == value
        
        return False
    
    # ============ Sorting & Ordering ============
    
    def order_by(self, *fields: str) -> 'DataSerials':
        """
        Return sorted copy of DataSerials.
        
        Args:
            *fields: Field names to sort by. Prefix with '-' for descending.
            
        Returns:
            New sorted DataSerials instance
        """
        if not fields:
            return self.copy()
        
        def sort_key(dp: DataPoint) -> Tuple:
            values = []
            for field in fields:
                if field.startswith('-'):
                    field_name = field[1:]
                    reverse = True
                else:
                    field_name = field
                    reverse = False
                
                value = dp.get(field_name)
                values.append((value, reverse))
            return tuple(v for v, _ in values)
        
        result = self.copy()
        result._data.sort(key=sort_key)
        
        # Handle descending sort
        for i, field in enumerate(fields):
            if field.startswith('-') and i == 0:
                result._data.reverse()
                break
        
        result._rebuild_index()
        return result
    
    def sort(self, by: str, reverse: bool = False) -> 'DataSerials':
        """Sort in-place by attribute and return self for chaining."""
        if not self._data:
            return self
        
        self._data.sort(key=lambda dp: dp.get(by), reverse=reverse)
        self._rebuild_index()
        return self
    
    # ============ Transformation ============
    
    def values(self, *fields: str) -> List[Dict[str, Any]]:
        """
        Return list of dictionaries with specified fields.
        
        Args:
            *fields: Field names to include
            
        Returns:
            List of dictionaries
        """
        return [
            {field: dp.get(field) for field in fields}
            for dp in self._data
        ]
    
    def values_list(self, *fields: str, flat: bool = False) -> List:
        """
        Return list of tuples with specified fields.
        
        Args:
            *fields: Field names to include
            flat: If True and only one field, return flat list
            
        Returns:
            List of tuples or flat list if flat=True
        """
        if flat and len(fields) == 1:
            return [dp.get(fields[0]) for dp in self._data]
        
        return [tuple(dp.get(f) for f in fields) for dp in self._data]
    
    def distinct(self, *fields: str) -> 'DataSerials':
        """
        Return distinct DataPoints based on specified fields.
        
        Args:
            *fields: Fields to check for distinctness. If empty, check by id.
            
        Returns:
            New DataSerials instance with distinct DataPoints
        """
        seen = set()
        result = self.__class__()
        result._metadata = copy.deepcopy(self._metadata)
        
        for dp in self._data:
            if not fields:
                key = dp.id
            else:
                key = tuple(dp.get(f) for f in fields)
            
            if key not in seen:
                seen.add(key)
                result.append(copy.deepcopy(dp), force_update=True)
        
        return result
    
    # ============ Aggregation ============
    
#     def aggregate(self, **kwargs) -> Dict[str, Any]:
#         """
#         Aggregate values across DataPoints.
        
#         Args:
#             **kwargs: Aggregation specifications (e.g., avg_age=('age', 'avg'))
            
#         Returns:
#             Dictionary with aggregation results
#         """
#         result = {}
        
#         for key, spec in kwargs.items():
#             if isinstance(spec, tuple) and len(spec) == 2:
#                 field, operation = spec
#                 values = [dp.get(field) for dp in self._data]
#                 values = [v for v in values if v is not None]
                
#                 if not values:
#                     result[key] = None
#                 elif operation == 'avg':
#                     result[key] = sum(values) / len(values)
#                 elif operation == 'sum':
#                     result[key] = sum(values)
#                 elif operation == 'max':
#                     result[key] = max(values)
#                 elif operation == 'min':
#                     result[key] = min(values)
#                 elif operation == 'count':
#                     result[key] = len(values)
#                 elif operation == 'count_distinct':
#                     result[key] = len(set(values))
#                 elif operation == 'list':
#                     result[key] = values
#                 elif operation == 'set':
#                     result[key] = set(values)
        
#         return result
    
    def count(self) -> int:
        """Return number of DataPoints."""
        return len(self._data)
    
    def exists(self) -> bool:
        """Check if collection has any DataPoints."""
        return bool(self._data)
    
    # ============ Statistics ============
    
    def describe(self, *fields: str) -> pd.DataFrame:
        """
        Generate descriptive statistics for numeric fields.
        
        Args:
            *fields: Fields to describe. If empty, all numeric fields.
            
        Returns:
            Pandas DataFrame with statistics
        """
        if not self._data:
            return pd.DataFrame()
        
        # Collect numeric fields
        if not fields:
            sample = self._data[0]
            fields = [k for k, v in sample._data.items() 
                     if isinstance(v, (int, float))]
        
        data = {field: [] for field in fields}
        for dp in self._data:
            for field in fields:
                data[field].append(dp.get(field))
        
        df = pd.DataFrame(data)
        return df.describe()
    
    # ============ Serialization ============
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'metadata': self._metadata,
            'data': [dp.to_dict() for dp in self._data],
            'created': datetime.now().isoformat(),
            'size': len(self._data)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSerials':
        """Create DataSerials from dictionary."""
        ds = cls()
        ds._metadata = copy.deepcopy(data.get('metadata', {}))
        
        for item in data.get('data', []):
            dp = DataPoint.from_dict(item)
            ds.append(dp, force_update=True)
        
        return ds
    
    def to_json(self, filepath: Optional[Union[str, Path]] = None, 
                **kwargs) -> Optional[str]:
        """
        Serialize to JSON.
        
        Args:
            filepath: Optional file path to save JSON
            **kwargs: Additional arguments for json.dump/dumps
            
        Returns:
            JSON string if filepath is None, else None
        """
        data = self.to_dict()
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, **kwargs)
            return None
        
        return json.dumps(data, indent=2, **kwargs)
    
    @classmethod
    def from_json(cls, filepath: Optional[Union[str, Path]] = None,
                 json_str: Optional[str] = None) -> 'DataSerials':
        """Create DataSerials from JSON."""
        if filepath and json_str:
            raise ValueError("Provide either filepath or json_str, not both")
        
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            if json_str is None:
                raise ValueError("Must provide either filepath or json_str")
            data = json.loads(json_str)
        
        return cls.from_dict(data)
    
    def to_xml(self, filepath: Optional[Union[str, Path]] = None, 
               pretty: bool = True, 
               encoding: str = 'unicode',
               xml_declaration: bool = True) -> Optional[Union[str, ET.Element]]:
        """
        Convert DataSerials to XML. 

        Args:
            filepath: Optional file path to save XML. If provided, saves to file and returns None.
            pretty: Whether to format XML for readability when saving to file.
            encoding: Output encoding ('unicode', 'utf-8', etc.)
            xml_declaration: Whether to include XML declaration in file output

        Returns:
            If filepath is None: XML string
            If filepath is provided: None (saves to file)
        """
        root = ET.Element('DataSerials')
        version = self.get_metadata('version', '1.0')
        root.set('version', version)
        root.set('created', datetime.now().isoformat())
        root.set('size', str(len(self._data)))

        # Add class information for potential versioning
        root.set('class', self.__class__.__name__)

        # Metadata
        if self._metadata:
            meta_elem = ET.SubElement(root, 'metadata')
            for key, value in self._metadata.items():
                attr_elem = ET.SubElement(meta_elem, 'attr', name=key)
                # Handle non-string values gracefully
                if value is None:
                    attr_elem.set('value', '')
                    attr_elem.set('type', 'none')
                elif isinstance(value, (bool, int, float)):
                    attr_elem.set('value', str(value))
                    attr_elem.set('type', type(value).__name__)
                elif isinstance(value, (list, tuple, dict)):
                    attr_elem.set('value', json.dumps(value))
                    attr_elem.set('type', 'json')
                else:
                    attr_elem.set('value', str(value))
                    attr_elem.set('type', 'str')

        # Data
        data_elem = ET.SubElement(root, 'data')
        for dp in self._data:
            dp_elem = dp.to_xml()  # Returns ET.Element
            data_elem.append(dp_elem)

        # If filepath provided, save to file
        if filepath is not None:
            # Create directory if it doesn't exist
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert to string
            if encoding == 'unicode':
                xml_str = ET.tostring(root, encoding='unicode')
            else:
                xml_str = ET.tostring(root, encoding=encoding).decode(encoding)

            # Pretty print
            if pretty:
                from xml.dom import minidom
                dom = minidom.parseString(xml_str if encoding == 'unicode' else xml_str.encode())
                xml_str = dom.toprettyxml(indent='  ')

                # Remove extra newline that minidom adds
                xml_str = '\n'.join(line for line in xml_str.split('\n') if line.strip())

            # Add XML declaration if requested and not already present
            if xml_declaration and not xml_str.startswith('<?xml'):
                declaration = f'<?xml version="1.0" encoding="{encoding if encoding != "unicode" else "utf-8"}"?>\n'
                xml_str = declaration + xml_str

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xml_str)

            return None

        # Otherwise return XML string
        if encoding == 'unicode':
            return ET.tostring(root, encoding='unicode')
        else:
            return ET.tostring(root, encoding=encoding).decode(encoding)


    @classmethod
    def from_xml(cls, source: Union[str, Path, bytes, ET.Element], **kwargs) -> 'DataSerials':
        """
        Create DataSerials from XML.

        Args:
            source: XML source - can be file path, XML string, bytes, or ET.Element
            **kwargs: Additional arguments for ET.parse/fromstring

        Returns:
            DataSerials instance

        Raises:
            ValueError: If XML format is invalid
            FileNotFoundError: If file path doesn't exist
        """
        # Handle different input types
        if isinstance(source, ET.Element):
            root = source

        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                # File path
                try:
                    tree = ET.parse(path, **kwargs)
                    root = tree.getroot()
                except ET.ParseError as e:
                    raise ValueError(f"Invalid XML file: {e}")
            else:
                # Try as XML string
                try:
                    root = ET.fromstring(source, **kwargs)
                except ET.ParseError as e:
                    if path.exists() is False:  # Re-raise with clearer message
                        raise FileNotFoundError(f"XML file not found: {path}")
                    raise ValueError(f"Invalid XML string: {e}")

        elif isinstance(source, bytes):
            # Bytes object
            try:
                root = ET.fromstring(source, **kwargs)
            except ET.ParseError as e:
                raise ValueError(f"Invalid XML bytes: {e}")

        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")

        # Validate root
        if root.tag != 'DataSerials':
            # Try to handle single DataPoint file
            if root.tag == 'DataPoint':
                ds = cls()
                dp = DataPoint.from_xml(root)
                ds.append(dp, force_update=True)
                return ds
            elif root.tag == 'DataPoints':
                ds = cls()
                for dp_elem in root.findall('DataPoint'):
                    dp = DataPoint.from_xml(dp_elem)
                    ds.append(dp, force_update=True)
                return ds
            else:
                raise ValueError(f"Invalid XML root: '{root.tag}'. Expected 'DataSerials', 'DataPoints', or 'DataPoint'")

        ds = cls()

        # Load metadata with type handling
        meta_elem = root.find('metadata')
        if meta_elem is not None:
            for attr_elem in meta_elem.findall('attr'):
                key = attr_elem.get('name')
                if not key:
                    continue

                value = attr_elem.get('value', '')
                value_type = attr_elem.get('type', 'str')

                # Convert value based on type
                if value_type == 'none':
                    ds.set_metadata(key, None)
                elif value_type == 'bool':
                    ds.set_metadata(key, value.lower() == 'true')
                elif value_type == 'int':
                    try:
                        ds.set_metadata(key, int(value))
                    except ValueError:
                        ds.set_metadata(key, 0)
                elif value_type == 'float':
                    try:
                        ds.set_metadata(key, float(value))
                    except ValueError:
                        ds.set_metadata(key, 0.0)
                elif value_type == 'json':
                    try:
                        ds.set_metadata(key, json.loads(value))
                    except json.JSONDecodeError:
                        ds.set_metadata(key, value)
                else:
                    ds.set_metadata(key, value)

        # Load data - handle different possible structures
        data_elem = root.find('data')
        if data_elem is not None:
            # New format: <data><DataPoint>...</DataPoint></data>
            for dp_elem in data_elem.findall('DataPoint'):
                try:
                    dp = DataPoint.from_xml(dp_elem)
                    ds.append(dp, force_update=True)
                except Exception as e:
                    warnings.warn(f"Failed to load DataPoint: {e}")
                    continue
        else:
            # Backward compatibility: direct DataPoint children
            for dp_elem in root.findall('DataPoint'):
                try:
                    dp = DataPoint.from_xml(dp_elem)
                    ds.append(dp, force_update=True)
                except Exception as e:
                    warnings.warn(f"Failed to load DataPoint: {e}")
                    continue

        return ds
    
    def to_pickle(self, filepath: Union[str, Path]) -> None:
        """Serialize to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> 'DataSerials':
        """Deserialize from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    # ============ Data Export ============
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = [dp.to_dict() for dp in self._data]
        return pd.DataFrame(records)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, id_column: str = 'id') -> 'DataSerials':
        """Create DataSerials from pandas DataFrame."""
        ds = cls()
        
        for _, row in df.iterrows():
            data = row.to_dict()
            dp_id = data.pop(id_column, None)
            dp = DataPoint(id=dp_id, **data)
            ds.append(dp, force_update=True)
        
        return ds
    
    # ============ Copy & Utility ============
    
    def copy(self, deep: bool = True) -> 'DataSerials':
        """Create a copy of the collection."""
        if deep:
            return copy.deepcopy(self)
        
        new_ds = self.__class__()
        new_ds._metadata = self._metadata.copy()
        new_ds._data = [dp.__copy__() for dp in self._data]
        new_ds._rebuild_index()
        return new_ds
    
    def head(self, n: int = 5) -> 'DataSerials':
        """Get first n DataPoints."""
        return self[:n]
    
    def tail(self, n: int = 5) -> 'DataSerials':
        """Get last n DataPoints."""
        return self[-n:] if n > 0 else self[0:0]
    
    def sample(self, n: Optional[int] = None, frac: Optional[float] = None,
               random_state: Optional[int] = None) -> 'DataSerials':
        """
        Get random sample of DataPoints.
        
        Args:
            n: Number of items to sample
            frac: Fraction of items to sample
            random_state: Random seed
            
        Returns:
            New DataSerials instance with sampled DataPoints
        """
        if n is None and frac is None:
            raise ValueError("Must provide either n or frac")
        
        if n is not None and frac is not None:
            raise ValueError("Provide either n or frac, not both")
        
        if n is not None:
            size = n
        else:
            size = int(len(self) * frac)
        
        if random_state is not None:
            random.seed(random_state)
        
        indices = random.sample(range(len(self._data)), min(size, len(self._data)))
        
        result = self.__class__()
        result._metadata = copy.deepcopy(self._metadata)
        
        for i in indices:
            result.append(copy.deepcopy(self._data[i]), force_update=True)
        
        return result
    
    def apply(self, func: Callable[[DataPoint], DataPoint]) -> 'DataSerials':
        """
        Apply a function to each DataPoint and return transformed collection.
        
        Args:
            func: Function that takes a DataPoint and returns a DataPoint
            
        Returns:
            New DataSerials instance with transformed DataPoints
        """
        result = self.__class__()
        result._metadata = copy.deepcopy(self._metadata)
        
        for dp in self._data:
            transformed = func(copy.deepcopy(dp))
            result.append(transformed, force_update=True)
        
        return result
    
    # ============ Magic Methods ============
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = [f"DataSerials(size={len(self)})"]
        
        if self._metadata:
            meta_str = ', '.join(f"{k}={repr(v)}" for k, v in self._metadata.items())
            parts.append(f"[{meta_str}]")
        
        if len(self._data) <= 10:
            for dp in self._data[:5]:
                parts.append(f"\n  {dp}")
        else:
            for dp in self._data[:3]:
                parts.append(f"\n  {dp}")
            parts.append("\n  ...")
            for dp in self._data[-3:]:
                parts.append(f"\n  {dp}")
        
        return ''.join(parts)
    
    def __str__(self) -> str:
        """Readable string representation."""
        return self.__repr__()
    
    def __bool__(self) -> bool:
        """Boolean value based on existence of data."""
        return bool(self._data)
    
    def group_by(self, *keys: str, **aggregations) -> 'DataSerials':
        """
        Group DataPoints by one or more fields and apply aggregations.

        Args:
            *keys: Field names to group by
            **aggregations: Aggregation specifications (field__agg=alias or (field, agg, alias))
                           Supported aggregations: count, sum, avg, mean, min, max, first, last, list, set, unique, std, var

        Returns:
            New DataSerials instance with one DataPoint per group

        Examples:
            # Basic grouping
            ds.group_by('category')

            # Group by multiple fields
            ds.group_by('category', 'status')

            # With aggregations
            ds.group_by('category', 
                        total=('price', 'sum'),
                        avg_price=('price', 'avg'),
                        count=('id', 'count'))

            # Shorthand syntax
            ds.group_by('category',
                        total='price__sum',
                        avg_price='price__avg')

            # Multiple aggregations on same field
            ds.group_by('category',
                        min_price=('price', 'min'),
                        max_price=('price', 'max'),
                        avg_price=('price', 'mean'))

            # Get lists/unique values
            ds.group_by('category',
                        all_names=('name', 'list'),
                        unique_status=('status', 'unique'))
        """
        if not keys:
            raise ValueError("At least one group by key must be specified")

        # Prepare aggregation specifications
        agg_specs = []

        # Process shorthand syntax (field__agg=alias)
        for alias, spec in aggregations.items():
            if isinstance(spec, str) and '__' in spec:
                field, agg = spec.rsplit('__', 1)
                agg_specs.append((field, agg, alias))
            elif isinstance(spec, tuple) and len(spec) in (2, 3):
                if len(spec) == 2:
                    field, agg = spec
                    agg_specs.append((field, agg, alias))
                else:
                    field, agg, custom_alias = spec
                    agg_specs.append((field, agg, custom_alias))
            else:
                raise ValueError(f"Invalid aggregation specification: {spec}")

        # If no aggregations specified, just count
        if not agg_specs:
            agg_specs.append(('id', 'count', 'count'))

        # Group data
        groups = defaultdict(list)

        for dp in self._data:
            # Create group key tuple
            group_key = tuple(dp.get(key) for key in keys)
            groups[group_key].append(dp)

        # Create result collection
        result = DataSerials()

        # Copy metadata
        result._metadata = copy.deepcopy(self._metadata)
        result.set_metadata('grouped_by', keys)
        result.set_metadata('group_count', len(groups))

        # Process each group
        for i, (group_key, group_dps) in enumerate(groups.items()):
            # Create DataPoint for this group
            group_dp = DataPoint()

            # Add group key fields
            for key, value in zip(keys, group_key):
                group_dp[key] = value

            # Add group_id
            group_dp['group_id'] = i
            group_dp['group_size'] = len(group_dps)

            # Apply aggregations
            for field, agg, alias in agg_specs:
                # Extract values, skipping None
                values = [dp.get(field) for dp in group_dps]

                # Filter out None for numeric operations
                if agg in ('sum', 'avg', 'mean', 'min', 'max', 'std', 'var'):
                    numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
                else:
                    numeric_values = values

                # Apply aggregation
                if agg == 'count':
                    # Count of non-null values
                    group_dp[alias] = len([v for v in values if v is not None])

                elif agg == 'count_distinct' or agg == 'nunique':
                    group_dp[alias] = len(set(v for v in values if v is not None))

                elif agg == 'sum':
                    group_dp[alias] = sum(numeric_values) if numeric_values else None

                elif agg in ('avg', 'mean'):
                    group_dp[alias] = sum(numeric_values) / len(numeric_values) if numeric_values else None

                elif agg == 'min':
                    group_dp[alias] = min(numeric_values) if numeric_values else None

                elif agg == 'max':
                    group_dp[alias] = max(numeric_values) if numeric_values else None

                elif agg == 'first':
                    group_dp[alias] = values[0] if values else None

                elif agg == 'last':
                    group_dp[alias] = values[-1] if values else None

                elif agg == 'list':
                    group_dp[alias] = [v for v in values if v is not None]

                elif agg == 'set' or agg == 'unique':
                    group_dp[alias] = list(set(v for v in values if v is not None))

                elif agg == 'std':
                    # Sample standard deviation
                    if len(numeric_values) > 1:
                        mean = sum(numeric_values) / len(numeric_values)
                        variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                        group_dp[alias] = variance ** 0.5
                    else:
                        group_dp[alias] = None

                elif agg == 'var':
                    # Sample variance
                    if len(numeric_values) > 1:
                        mean = sum(numeric_values) / len(numeric_values)
                        group_dp[alias] = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                    else:
                        group_dp[alias] = None

                elif agg == 'mode':
                    # Most frequent value
                    from collections import Counter
                    if values:
                        counter = Counter(v for v in values if v is not None)
                        if counter:
                            group_dp[alias] = counter.most_common(1)[0][0]
                        else:
                            group_dp[alias] = None
                    else:
                        group_dp[alias] = None

                elif agg == 'median':
                    if numeric_values:
                        sorted_vals = sorted(numeric_values)
                        n = len(sorted_vals)
                        if n % 2 == 0:
                            group_dp[alias] = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
                        else:
                            group_dp[alias] = sorted_vals[n//2]
                    else:
                        group_dp[alias] = None

                elif agg == 'concat':
                    # Concatenate strings
                    str_values = [str(v) for v in values if v is not None]
                    group_dp[alias] = ', '.join(str_values) if str_values else None

                else:
                    raise ValueError(f"Unsupported aggregation: {agg}")

            # Add to result
            result.append(group_dp, force_update=True)

        return result


    def pivot_table(self, index: Union[str, List[str]], columns: str, 
                    values: str, aggfunc: str = 'sum', 
                    fill_value: Any = None) -> pd.DataFrame:
        """
        Create a pivot table from the data.

        Args:
            index: Field(s) to use as row labels
            columns: Field to use as column labels
            values: Field to aggregate
            aggfunc: Aggregation function (sum, mean, count, etc.)
            fill_value: Value to fill for missing combinations

        Returns:
            Pandas DataFrame pivot table
        """
        if not self._data:
            return pd.DataFrame()

        # Collect data
        rows = []
        for dp in self._data:
            row = {}

            if isinstance(index, str):
                row['index'] = dp.get(index)
            else:
                for i, idx_field in enumerate(index):
                    row[f'index_{i}'] = dp.get(idx_field)

            row['columns'] = dp.get(columns)
            row['values'] = dp.get(values)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Create pivot table
        if isinstance(index, str):
            index_col = 'index'
        else:
            index_col = [f'index_{i}' for i in range(len(index))]

        pivot_df = df.pivot_table(
            index=index_col,
            columns='columns',
            values='values',
            aggfunc=aggfunc,
            fill_value=fill_value
        )

        return pivot_df


    def aggregate(self, **kwargs) -> Dict[str, Any]:
        """
        Aggregate values across all DataPoints.

        Args:
            **kwargs: Aggregation specifications

        Returns:
            Dictionary with aggregation results
        """
        result = {}

        for key, spec in kwargs.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                field, operation = spec
            elif isinstance(spec, str) and '__' in spec:
                field, operation = spec.rsplit('__', 1)
                key = key or f"{field}_{operation}"
            else:
                field = spec
                operation = 'count'

            values = [dp.get(field) for dp in self._data]
            # clean the values
            values = [v for v in values if v is not None]

            if not values:
                result[key] = None
            elif operation == 'count':
                result[key] = len(values)
            elif operation == 'count_distinct':
                result[key] = len(set(values))
            elif operation == 'sum':
                result[key] = sum(values)
            elif operation in ('avg', 'mean'):
                result[key] = sum(values) / len(values)
            elif operation == 'max':
                result[key] = max(values)
            elif operation == 'min':
                result[key] = min(values)
            elif operation == 'list':
                result[key] = values
            elif operation == 'set':
                result[key] = list(set(values))
            elif operation == 'first':
                result[key] = values[0]
            elif operation == 'last':
                result[key] = values[-1]
            elif operation == 'std':
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    result[key] = variance ** 0.5
                else:
                    result[key] = None
            elif operation == 'var':
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    result[key] = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                else:
                    result[key] = None

        return result


    def stats_by_group(self, group_field: str, value_field: str) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for each group.

        Args:
            group_field: Field to group by
            value_field: Numeric field to analyze

        Returns:
            Dictionary with group stats
        """
        groups = defaultdict(list)

        for dp in self._data:
            group = dp.get(group_field)
            value = dp.get(value_field)
            if value is not None and isinstance(value, (int, float)):
                groups[group].append(value)

        result = {}
        for group, values in groups.items():
            if values:
                result[group] = {
                    'count': len(values),
                    'sum': sum(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'median': sorted(values)[len(values)//2],
                    'std': (sum((x - sum(values)/len(values)) ** 2 for x in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else None
                }
            else:
                result[group] = {
                    'count': 0,
                    'sum': None,
                    'mean': None,
                    'min': None,
                    'max': None,
                    'median': None,
                    'std': None
                }

        return result

    def summary(self, detailed: bool = False, max_categories: int = 10) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.

        Args:
            detailed: If True, include detailed statistics for each field
            max_categories: Maximum number of unique values to show for categorical fields

        Returns:
            Dictionary containing dataset summary statistics

        Examples:
            # Basic summary
            summary = ds.summary()

            # Detailed summary with field-level statistics
            detailed_summary = ds.summary(detailed=True)

            # Print formatted summary
            ds.summary(print_report=True)
        """
        summary_dict = {
            'dataset': {
                'type': self.__class__.__name__,
                'size': len(self._data),
                'empty': len(self._data) == 0,
                'metadata_keys': list(self._metadata.keys()),
                'metadata_count': len(self._metadata),
            },
            'time': {
                'summary_generated': datetime.now().isoformat(),
            }
        }

        if not self._data:
            summary_dict['warning'] = 'Dataset is empty'
            return summary_dict

        # Collect all fields and their types
        all_fields = set()
        field_types = defaultdict(set)
        field_null_count = defaultdict(int)
        field_unique_values = defaultdict(set)
        field_values = defaultdict(list)

        for dp in self._data:
            for key, value in dp.items():
                if key == 'id':
                    continue
                all_fields.add(key)

                # Track value
                field_values[key].append(value)

                # Track nulls
                if value is None:
                    field_null_count[key] += 1

                # Track type
                if value is not None:
                    field_types[key].add(type(value).__name__)

                # Track unique values (limit for performance)
                if len(field_unique_values[key]) < max_categories * 2:
                    field_unique_values[key].add(value)

        # Dataset overview
        summary_dict['dataset'].update({
            'field_count': len(all_fields),
            'fields': sorted(all_fields),
            'has_id': any(dp.id is not None for dp in self._data),
            'id_coverage': f"{sum(1 for dp in self._data if dp.id is not None)}/{len(self._data)}",
        })

        # IDs summary if present
        ids = [dp.id for dp in self._data if dp.id is not None]
        if ids:
            summary_dict['ids'] = {
                'count': len(ids),
                'unique': len(set(ids)),
                'duplicates': len(ids) - len(set(ids)),
                'sample': ids[:5],
            }

        # Field summary
        field_summary = {}
        numeric_fields = []
        categorical_fields = []
        datetime_fields = []
        text_fields = []

        for field in sorted(all_fields):
            values = field_values[field]
            non_null_values = [v for v in values if v is not None]

            field_info = {
                'type': list(field_types[field]) if field_types[field] else ['unknown'],
                'null_count': field_null_count[field],
                'null_percentage': round(field_null_count[field] / len(values) * 100, 2),
                'unique_count': len(set(non_null_values)),
                'sample': non_null_values[:3] if non_null_values else None,
            }

            # Detect field category based on values
            if non_null_values:
                first_val = non_null_values[0]

                # Numeric
                if all(isinstance(v, (int, float)) for v in non_null_values[:100]):
                    numeric_fields.append(field)
                    numeric_vals = [v for v in non_null_values if isinstance(v, (int, float))]
                    field_info.update({
                        'category': 'numeric',
                        'min': min(numeric_vals),
                        'max': max(numeric_vals),
                        'mean': sum(numeric_vals) / len(numeric_vals),
                        'median': sorted(numeric_vals)[len(numeric_vals)//2],
                        'std': (sum((x - sum(numeric_vals)/len(numeric_vals)) ** 2 
                                   for x in numeric_vals) / (len(numeric_vals) - 1)) ** 0.5 
                               if len(numeric_vals) > 1 else None,
                        'zeros': sum(1 for v in numeric_vals if v == 0),
                        'negative': sum(1 for v in numeric_vals if v < 0),
                    })

                # Datetime
                elif any(isinstance(v, datetime) for v in non_null_values[:10]):
                    datetime_fields.append(field)
                    datetime_vals = [v for v in non_null_values if isinstance(v, datetime)]
                    field_info.update({
                        'category': 'datetime',
                        'min': min(datetime_vals).isoformat() if datetime_vals else None,
                        'max': max(datetime_vals).isoformat() if datetime_vals else None,
                        'range_days': (max(datetime_vals) - min(datetime_vals)).days 
                                     if len(datetime_vals) > 1 else 0,
                    })

                # Boolean
                elif all(isinstance(v, bool) for v in non_null_values[:100]):
                    categorical_fields.append(field)
                    bool_vals = [v for v in non_null_values if isinstance(v, bool)]
                    field_info.update({
                        'category': 'boolean',
                        'true_count': sum(1 for v in bool_vals if v),
                        'true_percentage': round(sum(1 for v in bool_vals if v) / len(bool_vals) * 100, 2),
                        'false_count': sum(1 for v in bool_vals if not v),
                    })

                # Categorical / String
                elif isinstance(first_val, str):
                    if field_info['unique_count'] < 20:  # Low cardinality
                        categorical_fields.append(field)
                        field_info.update({
                            'category': 'categorical',
                            'top_values': self._get_top_values(field_values[field], max_categories),
                        })
                    else:  # High cardinality / Text
                        text_fields.append(field)
                        field_info.update({
                            'category': 'text',
                            'avg_length': sum(len(str(v)) for v in non_null_values) / len(non_null_values),
                            'min_length': min(len(str(v)) for v in non_null_values),
                            'max_length': max(len(str(v)) for v in non_null_values),
                        })
                else:
                    categorical_fields.append(field)
                    field_info.update({
                        'category': 'other',
                        'top_values': self._get_top_values(field_values[field], max_categories),
                    })
            else:
                field_info['category'] = 'all_null'

            field_summary[field] = field_info

        summary_dict['fields'] = {
            'total': len(all_fields),
            'numeric': len(numeric_fields),
            'categorical': len(categorical_fields),
            'datetime': len(datetime_fields),
            'text': len(text_fields),
            'all_null': sum(1 for f in field_summary.values() 
                           if f.get('category') == 'all_null'),
        }

        # Add field details if requested
        if detailed:
            summary_dict['field_details'] = field_summary
            summary_dict['fields_by_category'] = {
                'numeric': numeric_fields,
                'categorical': categorical_fields,
                'datetime': datetime_fields,
                'text': text_fields,
            }

        # Correlation matrix for numeric fields (if detailed and multiple numeric fields)
        if detailed and len(numeric_fields) >= 2:
            try:
                import numpy as np
                # Build correlation matrix
                corr_matrix = {}
                for f1 in numeric_fields[:5]:  # Limit to first 5 for performance
                    corr_matrix[f1] = {}
                    v1 = [dp.get(f1) for dp in self._data 
                         if isinstance(dp.get(f1), (int, float))]

                    for f2 in numeric_fields[:5]:
                        if f1 == f2:
                            corr_matrix[f1][f2] = 1.0
                        else:
                            v2 = [dp.get(f2) for dp in self._data 
                                 if isinstance(dp.get(f2), (int, float))]

                            # Pair up values
                            pairs = []
                            for dp in self._data:
                                val1 = dp.get(f1)
                                val2 = dp.get(f2)
                                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                    pairs.append((val1, val2))

                            if len(pairs) > 1:
                                v1_vals, v2_vals = zip(*pairs)
                                corr = np.corrcoef(v1_vals, v2_vals)[0, 1]
                                corr_matrix[f1][f2] = round(corr, 3)
                            else:
                                corr_matrix[f1][f2] = None

                summary_dict['correlations'] = corr_matrix
            except ImportError:
                summary_dict['correlations'] = 'numpy not available'

        # Memory usage
        import sys
        summary_dict['performance'] = {
            'memory_usage_bytes': sys.getsizeof(self._data) + 
                                 sum(sys.getsizeof(dp) for dp in self._data),
            'data_points_memory': sum(sys.getsizeof(dp) for dp in self._data),
        }

        return summary_dict


    def _get_top_values(self, values: List[Any], n: int = 5) -> List[Dict]:
        """Get top n most frequent values and their counts."""
        from collections import Counter

        non_null = [v for v in values if v is not None]
        if not non_null:
            return []

        counter = Counter(non_null)
        total = len(non_null)

        top_values = []
        for value, count in counter.most_common(n):
            top_values.append({
                'value': value,
                'count': count,
                'percentage': round(count / total * 100, 2)
            })

        return top_values


    def summary_report(self, detailed: bool = False) -> str:
        """
        Generate a formatted text report of the dataset summary.

        Args:
            detailed: If True, include detailed field statistics

        Returns:
            Formatted string report
        """
        summary = self.summary(detailed=detailed)

        lines = []
        lines.append("=" * 80)
        lines.append(f"DATASET SUMMARY: {summary['dataset']['type']}")
        lines.append("=" * 80)

        # Basic info
        lines.append(f"\n📊 Basic Information:")
        lines.append(f"  • Size: {summary['dataset']['size']} data points")
        lines.append(f"  • Fields: {summary['dataset']['field_count']} total")
        lines.append(f"  • Metadata: {summary['dataset']['metadata_count']} attributes")
        if 'id_coverage' in summary['dataset']:
            lines.append(f"  • ID Coverage: {summary['dataset']['id_coverage']}")

        # Field composition
        fields = summary['fields']
        lines.append(f"\n🏷️  Field Composition:")
        lines.append(f"  • Numeric: {fields['numeric']} fields")
        lines.append(f"  • Categorical: {fields['categorical']} fields")
        lines.append(f"  • Datetime: {fields['datetime']} fields")
        lines.append(f"  • Text: {fields['text']} fields")
        if fields['all_null'] > 0:
            lines.append(f"  • ⚠️  All-Null: {fields['all_null']} fields")

        # Metadata
        if summary['dataset']['metadata_keys']:
            lines.append(f"\n📌 Metadata:")
            for key in summary['dataset']['metadata_keys'][:5]:
                value = self.get_metadata(key)
                lines.append(f"  • {key}: {value}")
            if len(summary['dataset']['metadata_keys']) > 5:
                lines.append(f"  • ... and {len(summary['dataset']['metadata_keys']) - 5} more")

        # Data quality
        lines.append(f"\n✅ Data Quality:")

        # Null analysis
        if detailed and 'field_details' in summary:
            null_fields = [(f, info['null_percentage']) 
                          for f, info in summary['field_details'].items()
                          if info['null_percentage'] > 0]
            null_fields.sort(key=lambda x: x[1], reverse=True)

            if null_fields:
                lines.append(f"  • Fields with missing values:")
                for field, pct in null_fields[:5]:
                    lines.append(f"    - {field}: {pct}% null")
                if len(null_fields) > 5:
                    lines.append(f"    - ... and {len(null_fields) - 5} more")
            else:
                lines.append(f"  • No missing values detected")

        # Numeric fields summary
        if detailed and 'field_details' in summary:
            numeric_infos = [(f, info) for f, info in summary['field_details'].items()
                            if info.get('category') == 'numeric']

            if numeric_infos:
                lines.append(f"\n📈 Numeric Fields Summary:")
                for field, info in numeric_infos[:5]:
                    lines.append(f"  • {field}:")
                    lines.append(f"    - Range: {info.get('min', 'N/A'):.2f} - {info.get('max', 'N/A'):.2f}")
                    lines.append(f"    - Mean: {info.get('mean', 'N/A'):.2f}")
                    lines.append(f"    - Std: {info.get('std', 'N/A'):.2f}")
                    lines.append(f"    - Null: {info['null_percentage']}%")

        # Categorical fields summary
        if detailed and 'field_details' in summary:
            cat_infos = [(f, info) for f, info in summary['field_details'].items()
                        if info.get('category') in ['categorical', 'boolean']]

            if cat_infos:
                lines.append(f"\n🔤 Categorical Fields Summary:")
                for field, info in cat_infos[:3]:
                    lines.append(f"  • {field}:")
                    lines.append(f"    - Unique: {info['unique_count']} values")
                    lines.append(f"    - Null: {info['null_percentage']}%")
                    if 'top_values' in info and info['top_values']:
                        lines.append(f"    - Top values:")
                        for tv in info['top_values'][:3]:
                            lines.append(f"      * {tv['value']}: {tv['count']} ({tv['percentage']}%)")

        # Memory usage
        if 'performance' in summary:
            mem_mb = summary['performance']['memory_usage_bytes'] / (1024 * 1024)
            lines.append(f"\n💾 Performance:")
            lines.append(f"  • Memory Usage: {mem_mb:.2f} MB")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


    def info(self) -> None:
        """
        Print concise dataset information (similar to pandas df.info()).
        """
        summary = self.summary(detailed=False)

        print(f"<{self.__class__.__name__}>")
        print(f"Size: {summary['dataset']['size']} entries")
        print(f"Fields: {summary['dataset']['field_count']} columns")

        if summary['dataset']['metadata_keys']:
            print(f"Metadata: {summary['dataset']['metadata_count']} attributes")

        print(f"\nData types:")

        if 'field_details' in summary:
            type_counts = {}
            for info in summary['field_details'].values():
                category = info.get('category', 'unknown')
                type_counts[category] = type_counts.get(category, 0) + 1

            for category, count in type_counts.items():
                print(f"  • {category}: {count}")

        print(f"\nMemory usage: {summary['performance']['memory_usage_bytes']:,} bytes")


    def describe(self, include: List[str] = None) -> pd.DataFrame:
        """
        Generate descriptive statistics (similar to pandas df.describe()).

        Args:
            include: List of field categories to include ('numeric', 'categorical', 'datetime')

        Returns:
            DataFrame with descriptive statistics
        """
        summary = self.summary(detailed=True)

        if 'field_details' not in summary:
            return pd.DataFrame()

        stats_data = {}

        for field, info in summary['field_details'].items():
            category = info.get('category')

            if include and category not in include:
                continue

            if category == 'numeric':
                stats_data[field] = {
                    'count': len(self._data) - info['null_count'],
                    'mean': info.get('mean'),
                    'std': info.get('std'),
                    'min': info.get('min'),
                    '25%': None,  # Would need percentile calculation
                    '50%': info.get('median'),
                    '75%': None,
                    'max': info.get('max'),
                    'null': f"{info['null_percentage']}%"
                }
            elif category in ['categorical', 'boolean']:
                stats_data[field] = {
                    'count': len(self._data) - info['null_count'],
                    'unique': info['unique_count'],
                    'top': info.get('top_values', [{}])[0].get('value') if info.get('top_values') else None,
                    'freq': info.get('top_values', [{}])[0].get('count') if info.get('top_values') else None,
                    'null': f"{info['null_percentage']}%"
                }
            elif category == 'datetime':
                stats_data[field] = {
                    'count': len(self._data) - info['null_count'],
                    'min': info.get('min'),
                    'max': info.get('max'),
                    'range_days': info.get('range_days'),
                    'null': f"{info['null_percentage']}%"
                }

        return pd.DataFrame(stats_data).T    
    
    
    def to_html_report(self, filepath: Optional[Union[str, Path]] = None, 
                       title: str = "DataSerials Report",
                       template: str = "modern",
                       include_plots: bool = True,
                       max_categories: int = 10) -> Optional[str]:
        """
        Generate an interactive HTML report of the dataset.

        Args:
            filepath: Optional file path to save HTML report
            title: Report title
            template: Report template style ('modern', 'simple', 'dark')
            include_plots: Whether to include matplotlib plots
            max_categories: Maximum categories to show in plots

        Returns:
            HTML string if filepath is None, else None
        """
        # Get summary data
        summary = self.summary(detailed=True)

        # Generate plots if requested
        plots_html = ""
        if include_plots and len(self._data) > 0:
            plots_html = self._generate_report_plots(max_categories)

        # Choose template
        if template == "simple":
            html = self._simple_html_template(summary, plots_html, title)
        elif template == "dark":
            html = self._dark_html_template(summary, plots_html, title)
        else:  # modern (default)
            html = self._modern_html_template(summary, plots_html, title)

        # Save or return
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            return None

        return html


    def _generate_report_plots(self, max_categories: int = 10) -> str:
        """Generate plots for HTML report."""
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        plots_html = ""

        if len(self._data) == 0:
            return ""

        # Collect field information
        numeric_fields = []
        categorical_fields = []
        datetime_fields = []

        for dp in self._data[:100]:  # Sample first 100 for type detection
            for key, value in dp.items():
                if key == 'id':
                    continue
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields.append(key)
                elif isinstance(value, str) and len(numeric_fields + categorical_fields) < 10:
                    if key not in categorical_fields and key not in numeric_fields:
                        categorical_fields.append(key)
                elif isinstance(value, datetime):
                    if key not in datetime_fields:
                        datetime_fields.append(key)

        # 1. Numeric fields distribution
        if numeric_fields:
            n_plots = min(4, len(numeric_fields))
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
            if n_plots == 1:
                axes = [axes]

            for i, field in enumerate(numeric_fields[:n_plots]):
                values = [dp.get(field) for dp in self._data 
                         if isinstance(dp.get(field), (int, float))]
                if values:
                    axes[i].hist(values, bins=20, edgecolor='black', alpha=0.7)
                    axes[i].set_title(f'{field} Distribution')
                    axes[i].set_xlabel(field)
                    axes[i].set_ylabel('Frequency')

            plt.tight_layout()

            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            plots_html += f"""
            <div class="plot-container">
                <h3>Numeric Field Distributions</h3>
                <img src="data:image/png;base64,{img_base64}" alt="Numeric Distributions">
            </div>
            """

        # 2. Categorical fields bar charts
        if categorical_fields:
            n_plots = min(4, len(categorical_fields))
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
            if n_plots == 1:
                axes = [axes]

            for i, field in enumerate(categorical_fields[:n_plots]):
                values = [dp.get(field) for dp in self._data if dp.get(field) is not None]
                if values:
                    from collections import Counter
                    counter = Counter(values)
                    top_items = dict(counter.most_common(max_categories))

                    axes[i].bar(range(len(top_items)), list(top_items.values()), 
                               tick_label=list(top_items.keys()))
                    axes[i].set_title(f'{field} Categories')
                    axes[i].set_xlabel(field)
                    axes[i].set_ylabel('Count')
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            plots_html += f"""
            <div class="plot-container">
                <h3>Categorical Field Distributions</h3>
                <img src="data:image/png;base64,{img_base64}" alt="Categorical Distributions">
            </div>
            """

        # 3. Correlation heatmap for numeric fields
        if len(numeric_fields) >= 2:
            try:
                import numpy as np

                # Build correlation matrix
                corr_data = []
                field_indices = {f: i for i, f in enumerate(numeric_fields[:6])}  # Max 6 fields

                for dp in self._data:
                    row = []
                    valid = True
                    for field in numeric_fields[:6]:
                        val = dp.get(field)
                        if isinstance(val, (int, float)):
                            row.append(val)
                        else:
                            valid = False
                            break
                    if valid and len(row) == len(field_indices):
                        corr_data.append(row)

                if len(corr_data) > 1:
                    corr_matrix = np.corrcoef(np.array(corr_data).T)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

                    # Add labels
                    ax.set_xticks(range(len(field_indices)))
                    ax.set_yticks(range(len(field_indices)))
                    ax.set_xticklabels(list(field_indices.keys()), rotation=45, ha='right')
                    ax.set_yticklabels(list(field_indices.keys()))

                    # Add correlation values
                    for i in range(len(field_indices)):
                        for j in range(len(field_indices)):
                            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                          ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')

                    plt.colorbar(im)
                    plt.title('Correlation Heatmap')
                    plt.tight_layout()

                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()

                    plots_html += f"""
                    <div class="plot-container">
                        <h3>Correlation Heatmap</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="Correlation Heatmap">
                    </div>
                    """
            except:
                pass

        return plots_html


    def _modern_html_template(self, summary: Dict, plots_html: str, title: str) -> str:
        """Modern HTML template with gradients and clean design."""

        # Format metadata
        metadata_html = ""
        if summary['dataset']['metadata_keys']:
            for key in summary['dataset']['metadata_keys']:
                value = self.get_metadata(key)
                metadata_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{key}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{value}</td>
                </tr>
                """

        # Format field summary
        field_summary_html = ""
        if 'field_details' in summary:
            for field, info in list(summary['field_details'].items())[:15]:  # Limit to 15 fields
                category = info.get('category', 'unknown')
                null_pct = info.get('null_percentage', 0)

                # Color coding for null percentage
                null_color = "#48bb78" if null_pct == 0 else "#ecc94b" if null_pct < 30 else "#f56565"

                field_summary_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;"><strong>{field}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">
                        <span style="background-color: #4299e1; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                            {category}
                        </span>
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{info.get('type', ['unknown'])[0]}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{info['unique_count']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">
                        <div style="background-color: #edf2f7; border-radius: 10px; height: 20px; width: 100%;">
                            <div style="background-color: {null_color}; width: {null_pct}%; height: 20px; border-radius: 10px;"></div>
                        </div>
                        <span style="font-size: 0.8em;">{null_pct}%</span>
                    </td>
                </tr>
                """

        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title} - DataSerials Report</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 40px 20px;
                }}

                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    overflow: hidden;
                }}

                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                }}

                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    font-weight: 300;
                }}

                .header h2 {{
                    font-size: 1.2em;
                    font-weight: 300;
                    opacity: 0.9;
                }}

                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 40px;
                    background: #f7fafc;
                }}

                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}

                .stat-card .value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 5px;
                }}

                .stat-card .label {{
                    color: #718096;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}

                .section {{
                    padding: 40px;
                    border-bottom: 1px solid #e2e8f0;
                }}

                .section h3 {{
                    font-size: 1.5em;
                    color: #2d3748;
                    margin-bottom: 20px;
                    font-weight: 400;
                }}

                .metadata-table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: #f7fafc;
                    border-radius: 12px;
                    overflow: hidden;
                }}

                .metadata-table th {{
                    text-align: left;
                    padding: 12px;
                    background: #edf2f7;
                    color: #4a5568;
                    font-weight: 600;
                }}

                .field-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                }}

                .field-table th {{
                    text-align: left;
                    padding: 12px;
                    background: #edf2f7;
                    color: #4a5568;
                    font-weight: 600;
                }}

                .plot-container {{
                    margin-bottom: 40px;
                    text-align: center;
                }}

                .plot-container img {{
                    max-width: 100%;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}

                .footer {{
                    background: #2d3748;
                    color: white;
                    padding: 20px 40px;
                    text-align: center;
                    font-size: 0.9em;
                    opacity: 0.8;
                }}

                .badge {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    font-weight: 600;
                }}

                .badge-success {{
                    background: #c6f6d5;
                    color: #22543d;
                }}

                .badge-warning {{
                    background: #feebc8;
                    color: #7b341e;
                }}

                .badge-danger {{
                    background: #fed7d7;
                    color: #742a2a;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <h2>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</h2>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['size']}</div>
                        <div class="label">Total DataPoints</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['field_count']}</div>
                        <div class="label">Fields</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['metadata_count']}</div>
                        <div class="label">Metadata Attributes</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{len(self._data) - sum(1 for dp in self._data if dp.id is None)}</div>
                        <div class="label">Entries with ID</div>
                    </div>
                </div>

                <div class="section">
                    <h3>📌 Metadata</h3>
                    <table class="metadata-table">
                        <tr>
                            <th>Attribute</th>
                            <th>Value</th>
                        </tr>
                        {metadata_html if metadata_html else '<tr><td colspan="2" style="padding: 20px; text-align: center; color: #718096;">No metadata available</td></tr>'}
                    </table>
                </div>

                <div class="section">
                    <h3>📊 Field Summary</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px;">
                        <div style="background: #ebf8ff; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.2em; font-weight: bold; color: #2b6cb0;">{summary['fields']['numeric']}</div>
                            <div style="color: #4a5568;">Numeric Fields</div>
                        </div>
                        <div style="background: #faf5ff; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.2em; font-weight: bold; color: #6b46c1;">{summary['fields']['categorical']}</div>
                            <div style="color: #4a5568;">Categorical Fields</div>
                        </div>
                        <div style="background: #fff5f5; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.2em; font-weight: bold; color: #c53030;">{summary['fields']['datetime']}</div>
                            <div style="color: #4a5568;">Datetime Fields</div>
                        </div>
                        <div style="background: #f0fff4; padding: 15px; border-radius: 8px;">
                            <div style="font-size: 1.2em; font-weight: bold; color: #276749;">{summary['fields']['text']}</div>
                            <div style="color: #4a5568;">Text Fields</div>
                        </div>
                    </div>

                    <table class="field-table">
                        <tr>
                            <th>Field</th>
                            <th>Category</th>
                            <th>Type</th>
                            <th>Unique</th>
                            <th>Null %</th>
                        </tr>
                        {field_summary_html}
                    </table>
                    {'' if len(summary.get('field_details', {})) <= 15 else f'<p style="text-align: right; color: #718096; margin-top: 10px;">Showing 15 of {len(summary["field_details"])} fields</p>'}
                </div>

                {f'''
                <div class="section">
                    <h3>📈 Visualizations</h3>
                    {plots_html}
                </div>
                ''' if plots_html else ''}

                <div class="footer">
                    Generated by DataSerials • {len(self._data)} data points • {summary['performance']['memory_usage_bytes'] / (1024*1024):.2f} MB
                </div>
            </div>
        </body>
        </html>
        """

        return html


    def _simple_html_template(self, summary: Dict, plots_html: str, title: str) -> str:
        """Simple, clean HTML template."""

        # Format field summary
        field_rows = ""
        if 'field_details' in summary:
            for field, info in summary['field_details'].items():
                field_rows += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">{field}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{info.get('category', 'unknown')}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{info['unique_count']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{info['null_count']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{info['null_percentage']}%</td>
                </tr>
                """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background: #f2f2f2; padding: 12px; text-align: left; }}
                td {{ padding: 8px; border: 1px solid #ddd; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: #f9f9f9; padding: 20px; border-radius: 5px; flex: 1; }}
                .plot {{ margin: 30px 0; text-align: center; }}
                .footer {{ margin-top: 50px; color: #999; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="stats">
                <div class="stat-box">
                    <strong>Total DataPoints:</strong> {summary['dataset']['size']}
                </div>
                <div class="stat-box">
                    <strong>Total Fields:</strong> {summary['dataset']['field_count']}
                </div>
                <div class="stat-box">
                    <strong>Metadata:</strong> {summary['dataset']['metadata_count']}
                </div>
            </div>

            <h2>Field Summary</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Category</th>
                    <th>Unique Values</th>
                    <th>Null Count</th>
                    <th>Null %</th>
                </tr>
                {field_rows}
            </table>

            {plots_html}

            <div class="footer">
                Memory Usage: {summary['performance']['memory_usage_bytes'] / (1024*1024):.2f} MB
            </div>
        </body>
        </html>
        """

        return html


    def _dark_html_template(self, summary: Dict, plots_html: str, title: str) -> str:
        """Dark mode HTML template."""

        # Format field summary
        field_rows = ""
        if 'field_details' in summary:
            for field, info in summary['field_details'].items():
                field_rows += f"""
                <tr style="border-bottom: 1px solid #404040;">
                    <td style="padding: 10px;">{field}</td>
                    <td style="padding: 10px;">{info.get('category', 'unknown')}</td>
                    <td style="padding: 10px;">{info['unique_count']}</td>
                    <td style="padding: 10px;">{info['null_count']}</td>
                    <td style="padding: 10px;">{info['null_percentage']}%</td>
                </tr>
                """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    background: #1a1a1a;
                    color: #e0e0e0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 40px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1 {{ color: #bb86fc; }}
                h2 {{ color: #03dac6; }}
                h3 {{ color: #cf6679; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: #2d2d2d;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th {{
                    background: #bb86fc;
                    color: #000;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #404040;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #bb86fc;
                }}
                .stat-card .value {{
                    font-size: 2em;
                    color: #bb86fc;
                    font-weight: bold;
                }}
                .plot-container {{
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .footer {{
                    margin-top: 50px;
                    padding: 20px;
                    text-align: center;
                    color: #666;
                    border-top: 1px solid #333;
                }}
                a {{ color: #bb86fc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['size']}</div>
                        <div>Total DataPoints</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['field_count']}</div>
                        <div>Fields</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{summary['dataset']['metadata_count']}</div>
                        <div>Metadata</div>
                    </div>
                </div>

                <h2>Field Details</h2>
                <table>
                    <tr>
                        <th>Field</th>
                        <th>Category</th>
                        <th>Unique</th>
                        <th>Null Count</th>
                        <th>Null %</th>
                    </tr>
                    {field_rows}
                </table>

                {plots_html}

                <div class="footer">
                    Memory: {summary['performance']['memory_usage_bytes'] / (1024*1024):.2f} MB
                </div>
            </div>
        </body>
        </html>
        """

        return html    
    
    # For backward compatibility
    query = filter
    filter_by = filter
    iloc = __getitem__
    loc = __getitem__
