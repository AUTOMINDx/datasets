#!/usr/bin/env python3
from datasets import load_dataset
from datasets import Dataset
from datasets import Features
from datasets import ClassLabel
from datasets import Value

# Load dataset from the Hugging Face Hub
dataset = load_dataset("lhoestq/demo1")

# Load dataset with a specific version
dataset = load_dataset("lhoestq/custom_squad", revision="main")

# Load a dataset without a loading script and map data files to splits
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)

# Load a specific subset of files using data_files or data_dir parameter
c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
c4_subset = load_dataset("allenai/c4", data_dir="en")

# Load a data file to a specific split
data_files = {"validation": "en/c4-validation.*.json.gz"}
c4_validation = load_dataset("allenai/c4", data_files=data_files, split="validation")

# Load a dataset using a local loading script
dataset = load_dataset("path/to/local/loading_script/loading_script.py", split="train")
dataset = load_dataset("path/to/local/loading_script", split="train")

# Edit a loading script and load it locally
eli5 = load_dataset("path/to/local/eli5")

# Load datasets from local and remote files
dataset = load_dataset("csv", data_files="my_file.csv")
dataset = load_dataset("json", data_files="my_file.json")
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
dataset = load_dataset("arrow", data_files={'train': 'train.arrow', 'test': 'test.arrow'})

# Load datasets from SQL databases
dataset = Dataset.from_sql("data_table_name", con="sqlite:///sqlite_file.db")
dataset = Dataset.from_sql("SELECT text FROM table WHERE length(text) > 100 LIMIT 10", con="sqlite:///sqlite_file.db")

# Load datasets using multiprocessing
oscar_afrikaans = load_dataset("oscar-corpus/OSCAR-2201", "af", num_proc=8)
imagenet = load_dataset("imagenet-1k", num_proc=8)
ml_librispeech_spanish = load_dataset("facebook/multilingual_librispeech", "spanish", num_proc=8)

# Load datasets from Python dictionaries
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# Load datasets from a list of dictionaries
my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
dataset = Dataset.from_list(my_list)

# Load datasets from a Python generator
def my_gen():
    for i in range(1, 4):
        yield {"a": i}

dataset = Dataset.from_generator(my_gen)

# Load datasets from Pandas DataFrames
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)

# Load datasets offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
dataset = load_dataset("my_dataset")

# Load a specific slice of a split
train_test_ds = datasets.load_dataset("bookcorpus", split="train+test")
train_10_20_ds = datasets.load_dataset("bookcorpus", split="train[10:20]")
train_10pct_ds = datasets.load_dataset("bookcorpus", split="train[:10%]")
train_10_80pct_ds = datasets.load_dataset("bookcorpus", split="train[:10%]+train[-80%:]")
val_ds = datasets.load_dataset("bookcorpus", split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
train_ds = datasets.load_dataset("bookcorpus", split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])

# Specify custom labels with Features
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
dataset = load_dataset('csv', data_files=file_dict, delimiter=';', column_names=['text', 'label'], features=emotion_features)

# Load a metric with load_metric()
from datasets import load_metric
metric = load_metric('PATH/TO/MY/METRIC/SCRIPT')

# Load configurations for a metric
metric = load_metric('bleurt', name='bleurt-base-128')
metric = load_metric('bleurt', name='bleurt-base-512')

# Load a metric in a distributed setup
from datasets import load_metric

# Assuming you are working in a distributed setup with multiple processes
num_process = 8  # Total number of processes
process_id = 3  # Process ID

metric = load_metric('glue', 'mrpc', num_process=num_process, process_id=process_id, experiment_id="My_experiment_10")

"""
In the above code, num_process represents the total number of processes running in the distributed setup, and process_id represents the unique ID assigned to each process. By providing the experiment_id, you can distinguish between multiple independent distributed evaluations running on the same server and files to avoid conflicts.

Please note that you need to replace 'glue' and 'mrpc' with the appropriate metric and task names you are working with in your specific scenario
"""
