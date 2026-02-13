"""Tests for DataSerials class."""

import pytest
from datamanagement import DataPoint, DataSerials


@pytest.fixture
def sample_ds():
    """Create a sample DataSerials instance."""
    ds = DataSerials()
    ds.append(DataPoint(id='1', name='Alice', age=25, score=85))
    ds.append(DataPoint(id='2', name='Bob', age=30, score=92))
    ds.append(DataPoint(id='3', name='Charlie', age=35, score=78))
    return ds


def test_filter_basic(sample_ds):
    """Test basic filtering."""
    result = sample_ds.filter(age__gt=28)
    assert len(result) == 2
    assert result[0].id == '2'
    assert result[1].id == '3'


def test_filter_multiple(sample_ds):
    """Test multiple conditions."""
    result = sample_ds.filter(age__gt=25, score__gte=80)
    assert len(result) == 2
    assert all(dp.score >= 80 for dp in result)
    assert all(dp.age > 25 for dp in result)


def test_group_by(sample_ds):
    """Test grouping functionality."""
    # Add more data for meaningful groups
    result = sample_ds.group_by('age_group',
                                avg_score=('score', 'mean'),
                                count=('id', 'count'))
    assert len(result) > 0
