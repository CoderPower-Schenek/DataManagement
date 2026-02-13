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
