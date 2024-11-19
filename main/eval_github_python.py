import os
import torch
import sys
import time
from transformers import pipeline
from tqdm import tqdm

from utils import load_json_from_file, validate_python_code, replace_key_in_json, write_json_to_file

FIXING_ATTEMPT_COUNT = 10

# Helper functions
def fix_code(pipe, instruction, code_snippet):
  code_error = validate_python_code(code_snippet)
  messages = [
    {"role": "system", "content": instruction}
  ]
  messages.append({"role": "user", "content": f"[Fix] | {code_error}\n{code_snippet}"})

  outputs = pipe(messages, max_new_tokens=512, pad_token_id=pipe.tokenizer.eos_token_id)

  return outputs[0]["generated_text"][-1]["content"]


def perform_fixing_code(pipe, instruction, dataset):
  results = []
  for item in tqdm(dataset, desc="Fixing code"):
    code_snippet = item['src']['string_format']

    fixing_attempts = []
    # Sub-progress bar for fixing attempts
    with tqdm(total=FIXING_ATTEMPT_COUNT, desc="Fixing attempts", leave=False) as pbar:
      for _ in range(FIXING_ATTEMPT_COUNT):
        fixed_code = fix_code(pipe, instruction, code_snippet)
        remain_error = validate_python_code(fixed_code)
        fixing_attempts.append({
          "string_format": fixed_code,
          "err_obj": 0 if remain_error is None \
            else {"msg": item["orig_err_obj"]["msg"], "msg_detailed": remain_error}
        })
        pbar.update(1)  # update progress bar

        # If the code is already fixed, no need to retry
        if remain_error is None:
          pbar.update(10)
          break

      pbar.close()  # close the progress bar after the loop
      # Update the return result
      results.append(replace_key_in_json(item, "pred", fixing_attempts))
  return results


################
# MAIN PROGRAM #
################

# Prepare token
torch.cuda.empty_cache()
hf_token = os.environ.get('HF_TOKEN')
print(hf_token)

# Load dataset
github_python_dataset = "../github-python-test"

# The original paper use the 3 4 as hold-out test set
test_dataset = []
test_dataset.append(load_json_from_file(os.path.join(github_python_dataset, 'model-fixer.pred.evaluated.3.json')))
test_dataset.append(load_json_from_file(os.path.join(github_python_dataset, 'model-fixer.pred.evaluated.4.json')))

# Prepare instruction
python_syntax_fixer_instruction = "You are an expert Python code fixer. \
             You will receive input in the following format: \n\n \
             [Fix] | <error code>\n \
             <python code snippet>\n\n \
             Your task is to ONLY provide the corrected Python code with NO explanations or additional text. \n \
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

# Record the start time
start_time = time.time()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")  # Print start time

# Perform fix
fixed_set_1 = perform_fixing_code(pipe, python_syntax_fixer_instruction, test_dataset[1])

# Write the result
write_json_to_file(
  fixed_set_1,
  os.path.join(github_python_dataset, 'model-fixer.pred.evaluated-llm.4.json'),
  2
)

# Record the end time
end_time = time.time()
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")  # Print end time

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Code fixing and writing took: {elapsed_time:.2f} seconds")
