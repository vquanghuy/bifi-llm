import json

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
with open("../pruto-deepfix-llm-checkpoint.4400_171124-0220.json", "r") as f:
  data = json.load(f)

# Summarize the successful fixes
summary = summarize_successful_fixes(data)

# Print the summary
print(summary)