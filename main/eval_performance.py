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

# Eval performance over 100 records, fix 1 time
github_python_dataset = "../github-python-test/model-fixer.pred.evaluated.3.json"

# Perform evaluate for 100 records using BIFI model
test_dataset = load_json_from_file(github_python_dataset)[:100]

# Perform evaluate for 100 records using BIFI model
# bifi_results = []
# for i, sample in enumerate(test_dataset):
#   start_time = time.time()
#   error_code = sample['src']['string_format']
#   fixed_code = perform_bifi_code_fix(error_code)
#   end_time = time.time()
#   elapsed_time = end_time - start_time
#
#   bifi_results.append({
#     'start_time': start_time,
#     'end_time': end_time,
#     'elapsed_time': elapsed_time,
#     'error_code': error_code,
#     'fixed_code': fixed_code,
#   })
#
#   # Write checkpoint
#   if (i + 1) % 20 == 0:
#     write_json_to_file(
#       bifi_results,
#       os.path.join(f'../eval-performance-bifi.checkpoint.{i}.json'),
#       2
#     )
#
# write_json_to_file(
#   bifi_results,
#   os.path.join(f'../eval-performance-bifi.json'),
#   2
# )

#############
# Eval using LLM
#############
# def perform_llm_fix_code(pipe, instruction, code_snippet):
#   code_error = validate_python_code(code_snippet)
#   messages = [
#     {"role": "system", "content": instruction}
#   ]
#   messages.append({"role": "user", "content": f"[Fix] | {code_error}\n{code_snippet}"})
#
#   outputs = pipe(messages, max_new_tokens=512, pad_token_id=pipe.tokenizer.eos_token_id)
#
#   return outputs[0]["generated_text"][-1]["content"]
#
# torch.cuda.empty_cache()
# hf_token = os.environ.get('HF_TOKEN')
#
# # Prepare instruction
# python_syntax_fixer_instruction = "You are an expert Python code fixer. \
#              You will receive input in the following format: \n\n \
#              [Fix] | <error code>\n \
#              <python code snippet>\n\n \
#              Your task is to ONLY provide the corrected Python code with NO explanations or additional text. \n \
#              Do not include the original error code in your response and do not format the code. \
#              Treat the code snippet as regular text. Do NOT put any prefix, only plain text as code only."
#
# # Load the model and instruction
# instruct_model_id = "meta-llama/Llama-3.2-3B-Instruct"
#
# pipe = pipeline(
#   "text-generation",
#   model=instruct_model_id,
#   token=hf_token,
#   torch_dtype=torch.bfloat16,
#   device_map="auto",
# )
#
# llm_results = []
# for i, sample in enumerate(test_dataset):
#   start_time = time.time()
#   error_code = sample['src']['string_format']
#   fixed_code = perform_llm_fix_code(pipe, python_syntax_fixer_instruction, error_code)
#   end_time = time.time()
#   elapsed_time = end_time - start_time
#
#   llm_results.append({
#     'start_time': start_time,
#     'end_time': end_time,
#     'elapsed_time': elapsed_time,
#     'error_code': error_code,
#     'fixed_code': fixed_code,
#   })
#
#   # Write checkpoint
#   if (i + 1) % 20 == 0:
#     write_json_to_file(
#       llm_results,
#       os.path.join(f'../eval-performance-llm.checkpoint.{i}.json'),
#       2
#     )
#
# write_json_to_file(
#   llm_results,
#   os.path.join(f'../eval-performance-llm.json'),
#   2
# )

# Summarize the result
bifi_result = load_json_from_file(os.path.join(f'../eval-performance-bifi.json'))
bifi_summary = {}

total_time = 0
for i in bifi_result:
  total_time += i['elapsed_time']
bifi_summary = {
  'total_sample': len(bifi_result),
  'avg_fix_time': total_time / len(bifi_result)
}
write_json_to_file(
  bifi_summary,
  os.path.join(f'../stats-performance-bifi.json'),
  2
)

llm_result = load_json_from_file(os.path.join(f'../eval-performance-llm.json'))
llm_summary = {}

total_time = 0
for i in llm_result:
  total_time += i['elapsed_time']
llm_summary = {
  'total_sample': len(llm_result),
  'avg_fix_time': total_time / len(llm_result)
}
write_json_to_file(
  llm_summary,
  os.path.join(f'../stats-performance-llm.json'),
  2
)



