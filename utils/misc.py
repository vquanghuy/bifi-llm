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
