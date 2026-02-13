# DataManagement Documentation
Welcome to DataManagement! This guide will help you understand and use the package effectively.

## 📦 Installation

```bash
# Install from PyPI
pip install datamanagement

# Or install from GitHub
pip install git+https://github.com/CoderPower-Schenek/DataManagement.git
```

## 🚀 Quick Start

```python
from datamanagement import DataPoint, DataSerials

# Create a collection
ds = DataSerials()

# Add some data points
ds.append(DataPoint(id='1', name='Alice', age=25, city='New York', score=85.5))
ds.append(DataPoint(id='2', name='Bob', age=30, city='London', score=92.0))
ds.append(DataPoint(id='3', name='Charlie', age=35, city='New York', score=78.5))
ds.append(DataPoint(id='4', name='Diana', age=28, city='Paris', score=95.0))

# Let's explore the data!
print(f"Total records: {len(ds)}")
```

## 📖 Core Concepts

### DataPoint
A flexible container for individual records. Think of it as a dictionary with superpowers.

```python
# Create a DataPoint
person = DataPoint(id='123', name='John', age=30, active=True)

# Access attributes (multiple ways)
print(person.name)        # Dot notation
print(person['age'])      # Dictionary notation
print(person.get('city', 'Unknown'))  # Safe access with default

# Update attributes
person.city = 'Boston'
person.update(age=31, active=False)

# Convert to dictionary
data_dict = person.to_dict()
```

### DataSerials
A collection of DataPoints with powerful query capabilities.

```python
# Create from list
data = [
    DataPoint(id='1', name='Alice', age=25),
    DataPoint(id='2', name='Bob', age=30)
]
ds = DataSerials(data)

# Or add one by one
ds.append(DataPoint(id='3', name='Charlie', age=35))
```

## 🔍 Querying Data

### Basic Filtering

```python
# Simple equality
young_people = ds.filter(age=25)
active_users = ds.filter(status='active')

# Multiple conditions
results = ds.filter(age=25, city='New York')
```

### Advanced Filtering with Operators

Use `field__operator=value` syntax:

```python
# Comparison operators
adults = ds.filter(age__gte=18)           # age >= 18
teens = ds.filter(age__lt=20)             # age < 20
not_adults = ds.filter(age__lte=17)        # age <= 17

# String operations
name_contains = ds.filter(name__contains='ob')     # name contains 'ob'
starts_with_a = ds.filter(name__startswith='A')    # name starts with 'A'
ends_with_n = ds.filter(name__endswith='n')        # name ends with 'n'

# List operations
in_cities = ds.filter(city__in=['New York', 'Boston'])  # city in list

# Null checking
missing_age = ds.filter(age__isnull=True)      # age is None
has_age = ds.filter(age__isnull=False)         # age is not None

# Range
age_range = ds.filter(age__between=[20, 30])   # 20 <= age <= 30

# Regular expressions
pattern_match = ds.filter(name__regex='^A.*')   # names starting with 'A'
```

### Custom Filter Functions

```python
# Use lambda functions for complex logic
high_scorers = ds.filter(lambda dp: dp.score > 90 and dp.age < 30)

# Define custom function
def is_valid(dp):
    return dp.age is not None and dp.score > 0

valid_records = ds.filter(is_valid)
```

### Chaining Filters

```python
# Method chaining works beautifully
results = (ds.filter(age__gte=18)
           .filter(city='New York')
           .filter(lambda dp: dp.score > 80))
```

## 📊 Data Analysis

### Aggregation

```python
# Single aggregations
avg_age = ds.aggregate(avg_age=('age', 'mean'))
total_score = ds.aggregate(total=('score', 'sum'))

# Multiple aggregations at once
stats = ds.aggregate(
    avg_age=('age', 'mean'),
    max_score=('score', 'max'),
    min_score=('score', 'min'),
    total_people=('id', 'count'),
    all_cities=('city', 'set')
)

print(stats)
# {
#     'avg_age': 29.5,
#     'max_score': 95.0,
#     'min_score': 78.5,
#     'total_people': 4,
#     'all_cities': {'New York', 'London', 'Paris'}
# }
```

### Group By Operations

```python
# Group by single field
by_city = ds.group_by('city')

# Group with aggregations
city_stats = ds.group_by('city',
    avg_age=('age', 'mean'),
    avg_score=('score', 'mean'),
    count=('id', 'count'),
    names=('name', 'list')
)

for group in city_stats:
    print(f"{group.city}: {group.count} people, avg score: {group.avg_score:.1f}")

# Group by multiple fields
by_city_age = ds.group_by('city', 'age_group',
    count=('id', 'count')
)
```

### Sorting

```python
# Sort by single field
sorted_by_age = ds.order_by('age')

# Sort descending
sorted_by_score_desc = ds.order_by('-score')

# Sort by multiple fields
sorted_complex = ds.order_by('city', '-age')
```

## 📈 Statistics and Summaries

### Quick Info
```python
ds.info()
# Output:
# <DataSerials>
# Size: 4 entries
# Fields: 5 columns
# Metadata: 0 attributes
#
# Data types:
#   • numeric: 2
#   • categorical: 2
#   • text: 1
#
# Memory usage: 2,456 bytes
```

### Summary Report
```python
# Basic summary
summary = ds.summary()

# Detailed summary with field statistics
detailed = ds.summary(detailed=True)

# Print formatted report
print(ds.summary_report())
```

### Statistical Description
```python
# Get statistics for numeric fields
desc_df = ds.describe()
print(desc_df)

# For specific field types
desc_df = ds.describe(include=['numeric', 'categorical'])
```

## 💾 Data Export/Import

### To/From Dictionary
```python
# Export
data_dict = ds.to_dict()

# Import
new_ds = DataSerials.from_dict(data_dict)
```

### To/From JSON
```python
# Save to file
ds.to_json('data.json')

# Load from file
ds = DataSerials.from_json('data.json')

# Get JSON string
json_str = ds.to_json()
```

### To/From XML
```python
# Save to file
ds.to_xml('data.xml', pretty=True)

# Load from file
ds = DataSerials.from_xml('data.xml')

# Get XML string
xml_str = ds.to_xml()
```

### To/From Pandas DataFrame
```python
import pandas as pd

# Convert to DataFrame
df = ds.to_dataframe()

# Create from DataFrame
new_ds = DataSerials.from_dataframe(df, id_column='id')
```

### To/From Pickle
```python
# Save
ds.to_pickle('data.pkl')

# Load
ds = DataSerials.from_pickle('data.pkl')
```

## 📊 Generate HTML Reports

Create beautiful, interactive reports:

```python
# Basic report
ds.to_html_report('report.html')

# Customize
ds.to_html_report(
    filepath='report.html',
    title='Customer Analysis Report',
    template='modern',      # 'modern', 'simple', or 'dark'
    include_plots=True
)
```

## 🎯 Practical Examples

### Example 1: Sales Data Analysis

```python
# Create sales data
sales = DataSerials()
sales.append(DataPoint(id='S1', product='Laptop', price=1200, quantity=2, region='US'))
sales.append(DataPoint(id='S2', product='Phone', price=800, quantity=5, region='EU'))
sales.append(DataPoint(id='S3', product='Tablet', price=500, quantity=3, region='US'))
sales.append(DataPoint(id='S4', product='Laptop', price=1200, quantity=1, region='Asia'))

# Calculate revenue for each sale
for sale in sales:
    sale.revenue = sale.price * sale.quantity

# Analyze by region
region_stats = sales.group_by('region',
    total_revenue=('revenue', 'sum'),
    avg_price=('price', 'mean'),
    total_items=('quantity', 'sum')
)

# Find high-value sales
high_value = sales.filter(revenue__gt=2000)

# Generate report
sales.to_html_report('sales_report.html', title='Sales Analysis')
```

### Example 2: User Activity Tracking

```python
# Create user activity data
from datetime import datetime, timedelta

activities = DataSerials()
activities.append(DataPoint(
    id='A1', 
    user_id='U1',
    action='login',
    timestamp=datetime.now() - timedelta(days=2),
    success=True
))
activities.append(DataPoint(
    id='A2',
    user_id='U1',
    action='purchase',
    timestamp=datetime.now() - timedelta(days=1),
    success=True,
    amount=150
))

# Filter recent activities
recent = activities.filter(
    timestamp__gte=datetime.now() - timedelta(days=7)
)

# Group by user
user_stats = activities.group_by('user_id',
    total_actions=('id', 'count'),
    total_spent=('amount', 'sum'),
    last_action=('timestamp', 'max')
)

# Find failed actions
failed = activities.filter(success=False)
```

## 🔧 Advanced Usage

### Custom Attributes on DataSerials

```python
ds = DataSerials()
ds.set_metadata('source', 'database')
ds.set_metadata('version', '1.0')
ds.set_metadata('created_by', 'admin')

# Access metadata
print(ds.get_metadata('source'))  # 'database'
```

### Slicing and Indexing

```python
# Access by position
first = ds[0]
last = ds[-1]

# Access by id
point = ds['123']

# Slice to create new collection
first_three = ds[:3]
every_other = ds[::2]

# Check existence
if '123' in ds:
    print("Found!")
```

### Iteration

```python
# Simple iteration
for dp in ds:
    print(f"{dp.id}: {dp.name}")

# Reverse iteration
for dp in reversed(ds):
    print(dp.name)

# With enumeration
for i, dp in enumerate(ds):
    print(f"{i}: {dp.name}")
```

### Sampling

```python
# Random sample
sample = ds.sample(n=5)

# Sample 20% of data
sample = ds.sample(frac=0.2)

# Reproducible sample
sample = ds.sample(n=5, random_state=42)
```

## ⚠️ Common Pitfalls & Tips

1. **Always set an ID**: DataPoints need IDs for dictionary-style access
   ```python
   # Good
   dp = DataPoint(id='123', name='Alice')
   
   # Bad - can't access by id
   dp = DataPoint(name='Alice')
   ```

2. **Force update when needed**: Use `force_update=True` to overwrite existing IDs
   ```python
   ds.append(new_dp, force_update=True)
   ```

3. **Check before filtering**: Empty filters return all data
   ```python
   # These are the same
   all_data = ds.filter()
   all_data = ds
   ```

4. **Memory efficiency**: Use `head()` and `tail()` for quick previews
   ```python
   ds.head(10)  # First 10 records
   ds.tail(5)   # Last 5 records
   ```

## 📚 API Reference

### DataPoint Methods

| Method | Description |
|--------|-------------|
| `get(key, default)` | Safe attribute access |
| `update(**kwargs)` | Update multiple attributes |
| `to_dict()` | Convert to dictionary |
| `to_xml()` | Convert to XML element |
| `keys()` | Get all attribute names |
| `values()` | Get all attribute values |
| `items()` | Get (key, value) pairs |

### DataSerials Methods

| Method | Description |
|--------|-------------|
| `filter(**kwargs)` | Filter data |
| `exclude(**kwargs)` | Exclude matching data |
| `group_by(*keys, **aggs)` | Group and aggregate |
| `order_by(*fields)` | Sort data |
| `aggregate(**kwargs)` | Calculate statistics |
| `describe()` | Statistical description |
| `summary()` | Dataset summary |
| `to_dataframe()` | Convert to pandas |
| `to_html_report()` | Generate HTML report |
| `head(n)` | First n records |
| `tail(n)` | Last n records |
| `sample(n)` | Random sample |

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/CoderPower-Schenek/DataManagement/issues)
- **Documentation**: [Read the Docs](https://datamanagement.readthedocs.io/)
- **Source Code**: [GitHub Repository](https://github.com/CoderPower-Schenek/DataManagement)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy coding! 🚀**
```markdown
This documentation is designed to be:
- **Simple and readable** - No complex jargon
- **Practical** - Lots of code examples users can copy-paste
- **Comprehensive** - Covers all major features
- **Well-organized** - Easy to find what you need

You can use this as your `README.md` or adapt it for your Sphinx documentation. Would you like me to also create the Sphinx configuration files to turn this into beautiful HTML documentation?
```
