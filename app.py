from flask import Flask, request
import sys
from pathlib import Path
import shutil
import os
import black
import ast
import gdown

# Set utils as searchable import
sys.path.insert(0, 'utils')

# Prepare variables
working_dir = Path('working')
model_dir = Path('models')
bifi_model = model_dir / 'bifi-fixer-round-2.pt'
token_vocab = working_dir / 'token-vocab.txt'

# Prepare working dir and model dir
if not working_dir.exists():
  working_dir.mkdir()

if not model_dir.exists():
  model_dir.mkdir()

os.environ["DATA_DIR"] = str(working_dir)

# Download the BIFI trained model
if not os.path.exists(bifi_model):
  gdown.download(id='1ZFdVEZhUkaO70IVxFhDTWfxrXPS5Dw2H', output=str(bifi_model), quiet=False)

# Download token vocab
if not os.path.exists(token_vocab):
  gdown.download(id='1Kp6m8BX4damo421fc-bJ27dflvyjsfE0', output=str(token_vocab), quiet=False)

# Import support packages
from utils.code_error_checker import check_paren_error, check_ast_error
from utils.code_utils import preprocess_unk, code_toks_to_code_string, tokenize_python_code
from utils.fairseq_utils import parse_fairseq_preds, fairseq_preprocess, fairseq_generate, fairseq_train

code_input = working_dir / 'code-input.txt'
token_input = working_dir / 'token-input.bad'
preprocess_dir = working_dir / 'preprocess'

model_dir = Path('models')
predict_path = working_dir / 'bifi-model.pred.txt'

def perform_bifi_code_fix(code_content):
  tokens, anonymize_dict = tokenize_python_code(code_content)

  # Preprocess code
  with open(str(token_input), 'w') as file:
    file.write(' '.join(tokens))
  shutil.rmtree(str(preprocess_dir))

  fairseq_preprocess(src='bad', tgt='good', workers=10,
                     destdir=str(preprocess_dir),
                     testpref=str(working_dir / 'token-input'),
                     srcdict=str(token_vocab),
                     only_source=True)
  shutil.copy(token_vocab, str(preprocess_dir / 'dict.good.txt'))

  # Perform code fix
  fairseq_generate(str(preprocess_dir), str(bifi_model), str(predict_path),
                   src='bad', tgt='good', gen_subset='test',
                   beam=10, nbest=1, max_len_a=1, max_len_b=50, max_tokens=7000)

  # Parse fixed code
  preds = parse_fairseq_preds(str(predict_path))

  # Convert to normal code
  predict_code = code_toks_to_code_string(preds[0]['pred'][0], anonymize_dict)

  return predict_code

#######################
# WEB SERVICE
#######################
app = Flask(__name__)


@app.route('/code-fixer', methods=['POST'])
def fix_code_endpoint():
  data = request.get_json()
  code = data['code']
  fixed_code = perform_bifi_code_fix(code)
  formatted_code = black.format_str(fixed_code, mode=black.FileMode())

  return {
    "fixed_code": formatted_code
  }


@app.route('/code-checker', methods=['POST'])
def check_code_endpoint():
  data = request.get_json()
  code = data['code']
  code_error = None

  try:
    ast.parse(code)
  except SyntaxError as e:
    code_error = e.msg
  finally:
    return {
      "has_error": code_error
    }


def create():
  return app


if __name__ == "__main__":
  app.run(debug=True, port=8080)
