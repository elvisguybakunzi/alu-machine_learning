# Pandas Project

## Learning Objectives

1. What is pandas?
   - Pandas is a powerful open-source data manipulation and analysis library for Python.
   - It provides high-performance, easy-to-use data structures and tools for working with structured data.

2. What is a pd.DataFrame? How do you create one?
   - A DataFrame is a 2-dimensional labeled data structure in pandas.
   - It can be thought of as a table or a spreadsheet-like structure.
   - You can create a DataFrame using various methods, such as:
     ```python
     import pandas as pd
     
     # From a dictionary
     df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
     
     # From a list of lists
     df = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']], columns=['A', 'B'])
     ```

3. What is a pd.Series? How do you create one?
   - A Series is a 1-dimensional labeled array in pandas.
   - It can hold data of any type (e.g., integers, floats, strings).
   - You can create a Series using:
     ```python
     import pandas as pd
     
     # From a list
     s = pd.Series([1, 2, 3, 4])
     
     # From a dictionary
     s = pd.Series({'a': 1, 'b': 2, 'c': 3})
     ```

4. How to load data from a file
   - Pandas provides various functions to read data from different file formats:
     ```python
     # CSV file
     df = pd.read_csv('file.csv')
     
     # Excel file
     df = pd.read_excel('file.xlsx')
     
     # JSON file
     df = pd.read_json('file.json')
     ```

5. How to perform indexing on a pd.DataFrame
   - You can access data in a DataFrame using various indexing methods:
     ```python
     # Column selection
     df['column_name']
     
     # Row selection by label
     df.loc[row_label]
     
     # Row selection by integer index
     df.iloc[row_index]
     ```

6. How to use hierarchical indexing with a pd.DataFrame
   - Hierarchical indexing (MultiIndex) allows you to have multiple levels of indexes:
     ```python
     df = pd.DataFrame(index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]],
                       columns=['X', 'Y'])
     ```

7. How to slice a pd.DataFrame
   - Slicing allows you to select a subset of data:
     ```python
     # Select rows 1 to 3 and columns 'A' to 'C'
     df.loc[1:3, 'A':'C']
     
     # Select first 5 rows and first 3 columns
     df.iloc[:5, :3]
     ```

8. How to reassign columns
   - You can rename columns using the `rename()` method:
     ```python
     df = df.rename(columns={'old_name': 'new_name'})
     ```

9. How to sort a pd.DataFrame
   - Use the `sort_values()` method to sort a DataFrame:
     ```python
     df = df.sort_values('column_name', ascending=True)
     ```

10. How to use boolean logic with a pd.DataFrame
    - You can filter data using boolean conditions:
      ```python
      # Select rows where column 'A' is greater than 5
      df[df['A'] > 5]
      ```

11. How to merge/concatenate/join pd.DataFrames
    - Pandas provides various methods for combining DataFrames:
      ```python
      # Merge two DataFrames
      pd.merge(df1, df2, on='key_column')
      
      # Concatenate DataFrames
      pd.concat([df1, df2])
      
      # Join DataFrames
      df1.join(df2)
      ```

12. How to get statistical information from a pd.DataFrame
    - Use the `describe()` method for summary statistics:
      ```python
      df.describe()
      ```

13. How to visualize a pd.DataFrame
    - Pandas integrates with matplotlib for basic plotting:
      ```python
      import matplotlib.pyplot as plt
      
      df.plot(kind='bar')
      plt.show()
      ```