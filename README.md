# DataManagement
DataManagement - A flexible data container with query capabilities.  

This pure-python package provides DataPoint and DataSerials classes for managing collections of data with powerful filtering, aggregation, and serialization.

## Features

- **Dynamic DataPoints**: Create objects with arbitrary attributes
- **Powerful Filtering**: Django-like query syntax (`age__gte=18`, `name__contains='John'`)
- **Aggregation**: Group by and aggregate data like SQL
- **Multiple Serialization**: XML, JSON, Pickle, Pandas DataFrame
- **Method Chaining**: Build complex queries elegantly
- **Rich Statistics**: Built-in summary, describe, and info methods
- **HTML Reports**: Generate beautiful interactive reports

## Installation

```bash
pip install datamanagement
```

## Quick Start
```python
from datamanagement import DataPoint, DataSerials

# Create data points
ds = DataSerials()
ds.append(DataPoint(id='1', name='Alice', age=25, score=85.5))
ds.append(DataPoint(id='2', name='Bob', age=30, score=92.0))

# Filter data
adults = ds.filter(age__gte=18, score__gte=80)

# Group and aggregate
stats = ds.group_by('age_group', 
                    avg_score=('score', 'mean'),
                    count=('id', 'count'))

# Generate report
ds.to_html_report('report.html')
```

## Documentation
Full documentation available at [Document.md](https://github.com/CoderPower-Schenek/DataManagement/blob/main/Document.md).

## License
MIT License - see [LICENSE](https://github.com/CoderPower-Schenek/DataManagement/blob/main/LICENSE) file for details.

