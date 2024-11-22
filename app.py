from flask import Flask, request
import sys
from pathlib import Path
import shutil
import os
import black
import ast
import gdown

from utils.misc import remove_backticks

# Set utils as searchable import
sys.path.insert(0, 'utils')

from utils.services import perform_bifi_code_fix, perform_llm_code_fix, cpp_syntax_fixer_instruction, \
  python_syntax_fixer_instruction
from utils.code_error_checker import validate_python_code, compile_code

#######################
# WEB SERVICE
#######################
app = Flask(__name__)


@app.route('/code-fixer', methods=['POST'])
def fix_code_endpoint():
  data = request.get_json()
  code = data['code']
  language = data.get('language', None)
  model = data.get('model', None)

  # Get input code
  if language == 'python':
    code_error = validate_python_code(code)
  elif language == 'cpp':
    code_error = compile_code(code)
  else:
    return {
      "error": "Unsupported language"
    }, 400  # Bad Request status code

  # Fixing code
  if language == 'python':
    if model == 'bifi':
      fixed_code = perform_bifi_code_fix(code)
    elif model == 'llm':
      fixed_code = perform_llm_code_fix(code, code_error, python_syntax_fixer_instruction)
    else:
      return {"error": "Unsupported model"}, 400  # Bad Request
  elif language == 'cpp':
    if model == 'llm':
      fixed_code = remove_backticks(perform_llm_code_fix(code, code_error, cpp_syntax_fixer_instruction))
    else:
      return {"error": "Unsupported model"}, 400  # Bad Request
  else:
    return {"error": "Unsupported language"}, 400  # Bad Request

  # Validate result
  if language == 'python':
    remain_error = validate_python_code(fixed_code)
  elif language == 'cpp':
    remain_error = compile_code(fixed_code)
  else:
    return {
      "error": "Unsupported language"
    }, 400  # Bad Request status code

  return {
    "code_error": code_error,
    "fixed_code": fixed_code,
    "remain_error": remain_error,
  }


@app.route('/code-checker', methods=['POST'])
def check_code_endpoint():
  data = request.get_json()
  code = data['code']
  language = data.get('language', None)
  code_error = None

  if language == 'python':
    code_error = validate_python_code(code)
  elif language == 'cpp':
    code_error = compile_code(code)
  else:
    return {
      "error": "Unsupported language"
    }, 400  # Bad Request status code

  return {
    "has_error": code_error
  }


def create():
  return app


if __name__ == "__main__":
  app.run(debug=True, port=5000)
