from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from enum import Enum


## vocabulary ##
def create_vocabulary(token_sentences: List[List[str]]) -> Set[str]:
  vocabulary = {token for tokens in token_sentences for token in tokens}
  vocabulary.add("[OOV]")
  return vocabulary

def create_token_to_id_mapping(vocabulary: Set[str]) -> Dict[str, int]:
  return {token: i for i, token in enumerate(vocabulary)}

def create_token_to_id_mapping_from_tokens(tokens: List[str]) -> Dict[str, int]:
  return create_token_to_id_mapping(create_vocabulary([tokens]))

def create_token_to_id_mapping_from_token_sentences(token_sentences: List[List[str]]) -> Dict[str, int]:
  return create_token_to_id_mapping(create_vocabulary(token_sentences))

## features ##
def get_bag_of_words_vector(tokens: List[str], token_to_id: Dict[str, int]) -> np.ndarray:
  bow = np.zeros(len(token_to_id))
  
  for token in tokens:
    if token in token_to_id:
        bow[token_to_id[token]] += 1
    else:
        bow[token_to_id["[OOV]"]] += 1
  
  return bow

def get_bag_of_words_vector_per_observation(dataset_column: pd.Series, token_to_id) -> np.ndarray:
  bow_per_observation = np.empty((len(dataset_column), len(token_to_id)))
  
  for i, _ in enumerate(dataset_column):
    tokens = dataset_column.iloc[i]
    bow_per_observation[i] = get_bag_of_words_vector(tokens, token_to_id)
    
  return bow_per_observation

def transform_bag_of_word_vectors_to_TF_idf_vectors(bag_of_word_vectors: np.ndarray) -> np.ndarray:
    tf_idf_transform = TfidfTransformer()
    return tf_idf_transform.fit_transform(bag_of_word_vectors).toarray() # BoW -> TF-IDF  

def count_tokens_per_observation(dataset_column: pd.Series):
  return np.array([len(tokens) for tokens in dataset_column])

def count_token_overlap_per_observation(dataset_column: pd.Series, reference_dataset_column: pd.Series, normalize: bool = True): 
  assert(dataset_column.shape[0] == reference_dataset_column.shape[0])
  
  token_overlap_per_obs = np.empty(dataset_column.shape[0])
  for i in range(dataset_column.shape[0]):
    tokens = dataset_column.iloc[i]
    reference_tokens = reference_dataset_column.iloc[i]
    
    token_to_ids = create_token_to_id_mapping_from_tokens(reference_tokens)
    token_overlap_per_obs[i] = np.sum([token in token_to_ids for token in tokens])
    
    if normalize:
      token_overlap_per_obs[i] /= len(tokens)
  
  return token_overlap_per_obs

def get_lexical_features_from_dataset(dataset: pd.DataFrame, token_to_id: Dict[str, int]) -> np.ndarray:
  questions = dataset['question']
  documents = dataset['document']

  if not isinstance(questions, pd.Series):
    raise TypeError("expected series")

  if not isinstance(documents, pd.Series):
    raise TypeError("expected series")

  question_BoW_per_obs         = get_bag_of_words_vector_per_observation(questions, token_to_id)
  question_token_count_per_obs = count_tokens_per_observation(questions)
  document_token_count_per_obs = count_tokens_per_observation(documents)
  token_overlap_per_obs        = count_token_overlap_per_observation(questions, documents)
  
  question_tf_idf_per_obs = transform_bag_of_word_vectors_to_TF_idf_vectors(question_BoW_per_obs)

  features = np.concatenate(
      (question_tf_idf_per_obs, 
       question_token_count_per_obs.reshape((-1,1)), 
       document_token_count_per_obs.reshape((-1,1)), 
       token_overlap_per_obs.reshape((-1,1))), 
      axis=1)

  return features

class Annotation_error(Enum):
    UNANSWERED = -1
    BAD_TOKENIZATION_OR_DATA = -2
    IGNORED = -3
    # Or the span e.g. (4,9)

def get_labels_from_dataset(dataset: pd.DataFrame) -> np.ndarray:
  answer_region_column = dataset['document_answer_region']

  if not isinstance(answer_region_column, pd.Series):
    raise TypeError("expected series")
 
  labels = np.empty(answer_region_column.shape[0], dtype=np.int32)
  for i, answer in enumerate(answer_region_column):
      labels[i] = 0 if answer == Annotation_error.UNANSWERED else 1
      
  return labels
