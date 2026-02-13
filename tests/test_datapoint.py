## Step 4: Write Tests

### `tests/test_datapoint.py`
```python
"""Tests for DataPoint class."""

import pytest
from datetime import datetime
from datamanagement import DataPoint


def test_datapoint_creation():
    """Test basic DataPoint creation."""
    dp = DataPoint(id='123', name='Test', value=42)
    assert dp.id == '123'
    assert dp.name == 'Test'
    assert dp.value == 42


def test_datapoint_dict_access():
    """Test dictionary-style access."""
    dp = DataPoint(id='123')
    dp['age'] = 25
    assert dp['age'] == 25
    assert dp.age == 25


def test_datapoint_serialization():
    """Test to_dict/from_dict."""
    original = DataPoint(id='123', name='Test', created=datetime.now())
    data = original.to_dict()
    restored = DataPoint.from_dict(data)
    assert restored.id == original.id
    assert restored.name == original.name
