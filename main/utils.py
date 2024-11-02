import json
import os
import ast

def load_json_from_file(json_file):
  """
  Loads JSON data from a file.

  Args:
    json_file: The path to the JSON file.

  Returns:
    The JSON data as a Python object (usually a dictionary or a list),
    or None if there was an error loading the file or decoding the JSON data.
  """
  try:
    with open(json_file, 'r') as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON in file {json_file}: {e}")  # Corrected line
    return None

  return data

def write_json_to_file(json_data, filename, indent=4):
  """
  Writes JSON data to a file.

  Args:
    json_data: The JSON data to write (can be a dictionary or a list).
    filename: The name of the file to write to.
    indent: (Optional) The number of spaces to use for indentation.
            Defaults to 4 for better readability.
  """

  with open(filename, 'w') as f:
    json.dump(json_data, f, indent=indent)

def replace_key_in_json(json_obj, key_to_replace, new_value):
  """
  Creates a new JSON object with a specific key replaced.

  Args:
    json_obj: The original JSON object.
    key_to_replace: The key to be replaced.
    new_value: The new value for the key.

  Returns:
    A new JSON object with the key replaced.
  """

  new_json_obj = json.loads(json.dumps(json_obj))  # Create a deep copy
  new_json_obj[key_to_replace] = new_value
  return new_json_obj

def validate_python_code(code_str):
  """
  Validates Python code and returns error information if any.

  Args:
    code_str: The Python code as a string.

  Returns:
    None if the code is valid, otherwise a string describing the syntax error.
  """
  try:
    ast.parse(code_str)
    return None  # No error
  except SyntaxError as e:
    return str(e)  # Return the error message as a string
