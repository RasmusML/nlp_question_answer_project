import pandas as pd

def save_dataset(df: pd.DataFrame, path: str):
    df.to_pickle(path)

def load_dataset(path: str):
  return pd.read_pickle(path)
  

def extract_data_by_language(dataset: pd.DataFrame, language: str):
  return dataset[dataset["language"] == language]

def load_datasets_by_language(training_set_path: str, validation_set_path: str):
  train_set = load_dataset(training_set_path)
  validation_set = load_dataset(validation_set_path)
  
  train_en = extract_data_by_language(train_set, "english")
  train_fi = extract_data_by_language(train_set, "finnish")
  train_ja = extract_data_by_language(train_set, "japanese")
  
  validation_en = extract_data_by_language(validation_set, "english")
  validation_fi = extract_data_by_language(validation_set, "finnish")
  validation_ja = extract_data_by_language(validation_set, "japanese")
  
  datasets = {
    "en": {"train_set": train_en, "validation_set": validation_en},
    "fi": {"train_set": train_fi, "validation_set": validation_fi},
    "ja": {"train_set": train_ja, "validation_set": validation_ja},
  }
  
  return datasets

