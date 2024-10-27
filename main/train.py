from datetime import datetime
import argparse
import os
import sys

sys.path.append("..")

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed, BitsAndBytesConfig
import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()
hf_token = os.environ.get('HF_TOKEN')

# transformers.logging.set_verbosity_info()
set_seed(42)
print("start time: ", get_current_time())

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-bs", "--batch-size", type=int, default=1)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-gcv", "--gradient-clip-val", type=float, default=0.0)
parser.add_argument("-wd", "--weight-decay", type=float, default=0)

parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-md", "--model-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-stl", "--save-total-limit", type=int, default=-1)
parser.add_argument("-pt", "--pre-trained", type=boolean_string, default=True)
args = parser.parse_args()

# Create job directory
model_name = "meta-llama/Llama-3.2-3B"
if args.model_dir != "":
    model_directory = args.model_dir
else:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_directory = model_name + "_global_" + dt_string

os.makedirs(model_directory)
with open(os.path.join(model_directory, "commandline_args.txt"), "w") as f:
    f.write("\n".join(sys.argv[1:]))

# Read and prepare data
data = GetDataAsPython("../data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython("../data/data_autofix_tracking_eslint_final.json")
data += data_eslint
all_warning_types = extract_warning_types(data)

if args.error_type != "":
    all_warning_types = [args.error_type]

print(all_warning_types)
(
    train_inputs,
    train_labels,
    val_inputs,
    val_labels,
    test_inputs,
    test_labels,
    train_info,
    val_info,
    test_info,
) = create_data(data, all_warning_types, include_warning=True, model_name=model_name)

# Load the  tokenizer using AutoClasses
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, device_map="cpu")
tokenizer.pad_token = tokenizer.eos_token

# Prepare BnB config for system optimization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare LoRA and apply to the model
lora_config = LoraConfig(
    r=8, # Rank of the LoRA update matrices
    lora_alpha=32, # Scaling factor for the LoRA update matrices
    lora_dropout=0.05, # Dropout probability for the LoRA update matrices
    bias="none", # Whether to apply a bias to the LoRA update matrices
    task_type="CAUSAL_LM" # Type of task for which to apply LoRA
)

model = get_peft_model(model, lora_config)

# Add special tokens to the tokenizer
# tokenizer.add_tokens(["{", "}", ">", "\\", "^"])
# tokenizer.save_pretrained(model_directory)

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model.resize_token_embeddings(len(tokenizer))
print("Models parameters: ", model.num_parameters())

# Create dataset required by pytorch
train_dataset = create_dataset(
    train_inputs, train_labels, tokenizer, pad_truncate=True, max_length=128
)
val_dataset = create_dataset(
    val_inputs, val_labels, tokenizer, pad_truncate=True, max_length=128
)

# Training arguments (adjust as needed) -> Error train
# training_args = TrainingArguments(
#     output_dir=model_directory,
#     num_train_epochs=args.epochs,
#     per_device_train_batch_size=args.batch_size,
#     per_device_eval_batch_size=args.batch_size,
#     warmup_steps=500,
#     weight_decay=args.weight_decay,
#     logging_dir=model_directory,
#     logging_steps=100,
#     do_eval=True,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     learning_rate=args.learning_rate,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=args.epochs if args.save_total_limit == -1 else args.save_total_limit,
#     eval_accumulation_steps=args.eval_acc_steps,
#     disable_tqdm=False,
#     seed=42,
# )

training_args = TrainingArguments(
    output_dir="./model-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    learning_rate=1e-4,
    fp16=True,  # use mixed precision if supported
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=args.learning_rate), None],
    tokenizer=tokenizer,
)

trainer.train()
print("end time: ", get_current_time())
