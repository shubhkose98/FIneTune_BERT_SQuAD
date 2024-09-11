# Fine-Tuning BERT for Question Answering on SQuAD Dataset
## Overview
This project explores how to fine-tune a BERT model for downstream tasks, specifically for Question Answering (QA) using the SQuAD dataset. The project compares the performance of a fine-tuned BERT model with a pre-finetuned DistilBERT model available on Hugging Face. Metrics like Exact Match (EM) and F1-score are used for evaluation.

## Implementation
- The model is based on BERT for QA and is initialized using BertForQuestionAnswering from the Hugging Face library.
- Fine-tuning involved adding additional layers on top of the BERT architecture for QA tasks.
- The Hugging Face Trainer class is used for training the model, while the model is evaluated using Exact Match (EM) and F1-score metrics.

## Dataset
The dataset used for this project is a subset of the Stanford Question Answering Dataset (SQuAD). SQuAD is a large-scale reading comprehension dataset with crowd-sourced questions based on Wikipedia articles. The answer to each question is either a segment of text (span) from the passage or the question is unanswerable. While SQuAD v1.1 contains over 100,000 question-answer pairs across 500+ articles, we have extracted a smaller subset of 10,000 samples for this fine-tuning task.

The extracted subset is split into three sets:

Train set: 70% of the samples
Evaluation set: 20% of the samples
Test set: 10% of the samples
This reduced dataset allows us to effectively fine-tune the BERT model without requiring extensive data for a downstream task like question-answering.

### Data Files
The provided dataset.zip file in the repo contains 6 pickle files, split into training, evaluation, and test datasets:
- *_dict.pkl: Contains raw data with context, questions, and answers.
- *_processed.pkl: Contains subword embeddings from the BERT tokenizer and embedder, along with offset mapping for the answer spans.

### Offset Mapping
Offset mapping is crucial for tasks like Question Answering, as it helps align tokenized words with their original character positions in the text. For example, given the sentence "Hello, world!" and its tokenized form ["Hello", ",", "world", "!"], the offset mapping would be [(0, 5), (6, 7), (8, 13), (13, 14)], mapping the token positions back to the original text.
In this project, offset mapping helps BERT predict the exact start and end token positions for the answer span within the passage.

## Training
- Model: BertForQuestionAnswering from Hugging Face's Transformers library.
- Model Initialization: BertForQuestionAnswering is used to set up the BERT-based model.
- Training Arguments: The training hyperparameters are configured, such as learning rate, batch size, etc.
- Training: The model is trained for 21 epochs which is the model with best producing results. 
- Optimizer: AdamW with appropriate learning rate decay.
- Evaluation: The model is evaluated using EM and F1-score.
- Comparison: Performance is compared to an open-source version of a pre-finetuned distilbert-base-cased-distilled-squad model from Hugging Face to benchmark performance.
  
## Requirements
- Hugging Face Transformers.
- Datasets, etc.
  
  ```pip install torch transformers datasets numpy tqdm ```

- Implementation details, required libraries and performance evaluation of the models can be found in the attached PDF.
- Please refer to the attached PDF file for further details on the outputs' implementation and visual representation and details of all the libraries and datasets required. Also, please check the imports in the .ipynb or .py file.
