import torch
import os
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime
import time
import json

from utils import write_json_to_file, compile_code, remove_backticks, get_error_code_snippets_from_db

#################
# MAIN
#################
FIXING_ATTEMPT_COUNT = 10

def fix_code(code_snippet):
  code_error = compile_code(code_snippet)
  messages = [
    {"role": "system", "content": cpp_syntax_fixer_instruction}
  ]
  messages.append({"role": "user", "content": f"[Fix] | {code_error}\n{code_snippet}"})

  outputs = pipe(messages, max_new_tokens=512, pad_token_id=pipe.tokenizer.eos_token_id)

  return outputs[0]["generated_text"][-1]["content"]

def perform_fixing_code(dataset):
  results = []
  for i, item in tqdm(enumerate(dataset), desc="Fixing code", total=len(dataset)):
    code_snippet = item[3]
    origin_error_message = item[4]

    repair_item = {
      "id": i,
      "error_code": code_snippet,
      "error_message": origin_error_message,
    }

    fixing_attempts = []
    # Sub-progress bar for fixing attempts
    with tqdm(total=FIXING_ATTEMPT_COUNT, desc="Fixing attempts", leave=False) as pbar:
      for _ in range(FIXING_ATTEMPT_COUNT):
        fixed_code = fix_code(code_snippet)
        # Remove backticks
        fixed_code = remove_backticks(fixed_code)
        remain_error = compile_code(fixed_code)
        fixing_attempts.append({
          "id": i + 1,
          "fixed_code": fixed_code,
          "error_message": '' if remain_error is None else remain_error
        })
        pbar.update(1)  # update progress bar

        # If the code is already fixed, no need to retry
        if remain_error is None:
          pbar.update(10)
          break

      pbar.close()  # close the progress bar after the loop
      # Update the return result
      repair_item['attempts'] = fixing_attempts
      results.append(repair_item)

    # Write checkpoint every 10 items
    if (i + 1) % 100 == 0:  # Check if 'i + 1' is divisible by 10
      timestamp = datetime.now().strftime("%d%m%y-%H%M")
      write_json_to_file(
        results,
        os.path.join(f'../pruto-deepfix-llm-checkpoint.{i + 1}_{timestamp}.json'),  # Use 'i + 1' for checkpoint number
        2
      )
  return results

# Prepare token
torch.cuda.empty_cache()
hf_token = os.environ.get('HF_TOKEN')

prutor_deepfix_dataset_db = '../prutor-deepfix-09-12-2017/prutor-deepfix-09-12-2017.db'

# Load dataset
prutor_deepfix_dataset = get_error_code_snippets_from_db(prutor_deepfix_dataset_db)

# Prepare instruction
cpp_syntax_fixer_instruction = "You are an expert C/C++ code fixer. \
             You will receive input in the following format: \n\n \
             [Fix] | <error code>\n \
             <code snippet>\n\n \
             Your task is to ONLY provide the corrected C/C++ code with NO explanations or additional text. \n \
             Do not include the original error code in your response and do not format the code. \
             Treat the code snippet as regular text. Do NOT put any prefix, only plain text as code only."

# Load the model and instruction
instruct_model_id = "meta-llama/Llama-3.2-3B-Instruct"

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
  "text-generation",
  model=model_id,
  token=hf_token,
  torch_dtype=torch.bfloat16,
  device_map="auto",
)

# Record the start time
start_time = time.time()

# Perform fix on dataset
fixed_set = perform_fixing_code(prutor_deepfix_dataset)

# Write the result to files
timestamp = datetime.now().strftime("%d%m%y-%H%M")
write_json_to_file(
  fixed_set,
  os.path.join(f'../pruto-deepfix-llm.{timestamp}.json'),
  2
)

# Record the end time
end_time = time.time()
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")  # Print end time

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Code fixing and writing took: {elapsed_time:.2f} seconds")

# Summarize the result
def summarize_successful_fixes(data):
  """
  Summarizes the successful fixes from a JSON array.

  Args:
      data: A JSON array, where each object represents a code fixing attempt.

  Returns:
      A dictionary containing:
          - total_fixes: The total number of code fixes.
          - successful_fixes: The number of successful fixes.
  """
  total_fixes = 0
  successful_fixes = 0

  for item in data:
    total_fixes += 1
    for attempt in item["attempts"]:
      if not attempt["error_message"]:
        successful_fixes += 1
        break

  success_rate = (successful_fixes / total_fixes) * 100 if total_fixes else 0
  summary = {
    "total_fixes": total_fixes,
    "successful_fixes": successful_fixes,
    "success_rate": success_rate,
  }

  return summary

# Load the data from the JSON file
with open("../pruto-deepfix-llm-checkpoint.6900_171124-0939.json", "r") as f:
  data = json.load(f)

# Summarize the successful fixes
summary = summarize_successful_fixes(data)

# Print the summary
print(summary)
