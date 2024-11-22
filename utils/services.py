import sys
from pathlib import Path
import shutil
import os
import gdown
import time
from transformers import pipeline
import torch

# Set utils as searchable import
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Insert the upper directory into the Python path
sys.path.insert(0, utils_dir)

# Prepare variables
working_dir = Path('../working')
data_dir = Path('../data')
model_dir = Path('../models')
preprocess_dir = working_dir / 'preprocess'
bifi_model = model_dir / 'bifi-fixer-round-2.pt'
token_vocab = data_dir / 'token-vocab.txt'

os.environ["DATA_DIR"] = str(data_dir)

# Prepare directories
if not data_dir.exists():
  data_dir.mkdir()

if not working_dir.exists():
  working_dir.mkdir()

if not model_dir.exists():
  model_dir.mkdir()

if not preprocess_dir.exists():
  preprocess_dir.mkdir()

# Download the BIFI trained model
if not os.path.exists(bifi_model):
  gdown.download(id='1ZFdVEZhUkaO70IVxFhDTWfxrXPS5Dw2H', output=str(bifi_model), quiet=False)

# Download token vocab and dict.good.txt
if not os.path.exists(token_vocab):
  gdown.download(id='1Kp6m8BX4damo421fc-bJ27dflvyjsfE0', output=str(token_vocab), quiet=False)

# Import support packages
from utils.code_utils import preprocess_unk, code_toks_to_code_string, tokenize_python_code
from utils.fairseq_utils import parse_fairseq_preds, fairseq_preprocess, fairseq_generate
from utils.json_utils import load_json_from_file, write_json_to_file
from utils.code_error_checker import validate_python_code

code_input = working_dir / 'code-input.txt'
token_input = working_dir / 'token-input.bad'

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


torch.cuda.empty_cache()
hf_token = os.environ.get('HF_TOKEN')

# Prepare instruction
python_syntax_fixer_instruction = "You are an expert Python code fixer. \
             You will receive input in the following format: \n\n \
             [Fix] | <error code>\n \
             <python code snippet>\n\n \
             Your task is to ONLY provide the corrected Python code with NO explanations or additional text. \n \
             Do not include the original error code in your response and do not format the code. \
             Treat the code snippet as regular text. Do NOT put any prefix, only plain text as code only."

cpp_syntax_fixer_instruction = "You are an expert C/C++ code fixer. \
             You will receive input in the following format: \n\n \
             [Fix] | <error code>\n \
             <code snippet>\n\n \
             Your task is to ONLY provide the corrected C/C++ code with NO explanations or additional text. \n \
             Do not include the original error code in your response and do not format the code. \
             Treat the code snippet as regular text. Do NOT put any prefix, only plain text as code only."

# Load the model and instruction
instruct_model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipe = pipeline(
  "text-generation",
  model=instruct_model_id,
  token=hf_token,
  torch_dtype=torch.bfloat16,
  device_map="auto",
)


def perform_llm_code_fix(code_snippet, code_error, instruction):
  messages = [
    {"role": "system", "content": instruction}
  ]
  messages.append({"role": "user", "content": f"[Fix] | {code_error}\n\n{code_snippet}"})

  outputs = pipe(messages, max_new_tokens=512, pad_token_id=pipe.tokenizer.eos_token_id)

  return outputs[0]["generated_text"][-1]["content"]
