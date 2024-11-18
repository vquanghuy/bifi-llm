from flask import Flask, request
import sys
from pathlib import Path
import shutil
import os
import black
import ast

# Set utils as searchable import
sys.path.insert(0, 'utils')

# Set environment variable
DATA_DIR = 'data'
os.environ["DATA_DIR"] = DATA_DIR

# Import support packages
from utils.code_error_checker import check_paren_error, check_ast_error
from utils.code_utils import preprocess_unk, code_toks_to_code_string, get_diff_metric, tokenize_python_code
from utils.fairseq_utils import parse_fairseq_preds, fairseq_preprocess, fairseq_generate, fairseq_train

working_dir = Path('data-apply')
code_input = working_dir/'code-input.txt'
token_input = working_dir/'token-input.bad'
token_vocab = working_dir/'token-vocab.txt'

preprocess_dir = working_dir/'preprocess'

model_dir = Path('models')
predict_model = model_dir/'round2-BIFI-part2-checkpoint.pt'
predict_path = working_dir/'bifi-model.pred.txt'

def perform_code_fix(code_content):
    tokens, anonymize_dict = tokenize_python_code(code_content)

    # Preprocess code
    with open(str(token_input), 'w') as file:
        file.write(' '.join(tokens))
    shutil.rmtree(str(preprocess_dir))

    fairseq_preprocess(src='bad', tgt='good', workers=10,
                       destdir  = str(preprocess_dir),
                       testpref = str(working_dir/'token-input'),
                       srcdict = str(token_vocab),
                       only_source=True)
    shutil.copy(token_vocab, str(preprocess_dir/'dict.good.txt'))

    # Perform code fix
    fairseq_generate(str(preprocess_dir), str(predict_model), str(predict_path),
                     src='bad', tgt='good', gen_subset='test',
                     beam=10, nbest=10, max_len_a=1, max_len_b=50, max_tokens=7000)

    # Parse fixed code
    preds = parse_fairseq_preds(str(predict_path))

    # Convert to normal code
    predict_code = code_toks_to_code_string(preds[0]['pred'][0], anonymize_dict)

    return predict_code
app = Flask(__name__)

@app.route('/code-fixer', methods=['POST'])
def fix_code_endpoint():
    data = request.get_json()
    code = data['code']
    fixed_code = perform_code_fix(code)
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