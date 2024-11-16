import json
import os
import ast
import subprocess

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


def get_error_code_snippets_from_db(db_path):
  """
  Retrieves all code that contains error in the SQLite database.

  Args:
  db_path: Path to the SQLite database file.

  Returns:
  A list of all code contains error.
  """
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  try:
    cursor.execute("SELECT * FROM Code WHERE errorcount != 0")
    data = [row for row in cursor.fetchall()]
    return data
  finally:
    conn.close()


def compile_code(code):
  """
  Compiles the given C code using gcc and returns the error message
  if compilation fails.

  Args:
    code: The C code as a string.

  Returns:
    An empty string if compilation is successful,
    the error message otherwise.
  """
  try:
    with open('temp.c', 'w') as f:
      f.write(code)

    result = subprocess.run(['gcc', 'temp.c', '-o', 'temp'],
                            capture_output=True, text=True)

    if result.returncode == 0:
      return None  # No error message
    else:
      return result.stderr  # Return the error message

  finally:
    import os
    try:
      os.remove('temp.c')
      os.remove('temp')
    except OSError:
      pass


import re


def remove_backticks(text):
  """
  Removes backtick code formatting from a string.

  Args:
    text: The string containing code blocks enclosed in backticks.

  Returns:
    The string with backtick code formatting removed.
  """
  pattern = r'```(?:[a-z]+)?\n(.*?)```'  # Matches code blocks with optional language
  return re.sub(pattern, r'\1', text, flags=re.DOTALL)
