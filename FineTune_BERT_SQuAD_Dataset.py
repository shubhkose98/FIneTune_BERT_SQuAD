#This code is used from the Homework 10 Lab manual document with my own modifications.

!pip install transformers[torch]

!pip install transformers

!pip install datasets

import pickle

with open('/content/drive/MyDrive/dataset/train_dict.pkl', 'rb') as f:
    train_dict = pickle.load(f)
with open('/content/drive/MyDrive/dataset/test_dict.pkl', 'rb') as f:
    test_dict = pickle.load(f)
with open('/content/drive/MyDrive/dataset/eval_dict.pkl', 'rb') as f:
    eval_dict = pickle.load(f)

with open('/content/drive/MyDrive/dataset/train_data_processed.pkl', 'rb') as f:
    train_processed = pickle.load(f)
with open('/content/drive/MyDrive/dataset/test_data_processed.pkl', 'rb') as f:
    test_processed = pickle.load(f)
with open('/content/drive/MyDrive/dataset/eval_data_processed.pkl', 'rb') as f:
    eval_processed = pickle.load(f)

print(train_dict.keys())
print(test_dict.keys())
print(eval_dict.keys())

print(train_processed.keys())
print(test_processed.keys())
print(eval_processed.keys())

from transformers import BertForQuestionAnswering

model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)

print(model._modules)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=21,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
)




from transformers import Trainer
from datasets import Dataset
import pandas as pd
from transformers import TrainerCallback
from transformers.trainer_utils import IntervalStrategy
import numpy as np



train_dataset = Dataset.from_pandas(pd.DataFrame(train_processed))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_processed))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_processed))


# Custom callback class
class EpochLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        logs = state.log_history[-1]  # Get the latest logged metrics

        # Extract the loss and learning rate from the Trainer
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")

        # Print the loss, learning rate, and epoch information
        print(f"{{'loss': {loss}, 'learning_rate': {learning_rate}, 'epoch': {epoch}}}")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Attach the custom logging callback to the Trainer to print after each epoch
trainer.add_callback(EpochLoggingCallback())

# Train the model
trainer.train()



import numpy as np

def compute_exact_match(prediction, truth):
    return int(prediction == truth)

def f1_score(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    # if either the prediction or the truth is no-answer then F1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then F1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


x = trainer.predict(test_dataset)
start_pos, end_pos = x.predictions
start_pos = np.argmax(start_pos, axis=1)
end_pos = np.argmax(end_pos, axis=1)

# Initialize lists to store EM and F1 scores
em_scores = []
f1_scores = []

# Iterate over predictions
for k, (i, j) in enumerate(zip(start_pos, end_pos)):
    # Get tokens for the current prediction
    tokens = tokenizer.convert_ids_to_tokens(test_processed['input_ids'][k])

    # Convert token list to string
    predicted_answer = tokenizer.convert_tokens_to_string(tokens[i:j+1]).lower()
    correct_answer = test_dict['answers'][k]['text'][0].lower()

    # Compute EM and F1 scores
    em = compute_exact_match(predicted_answer, correct_answer)
    f1 = f1_score(predicted_answer, correct_answer)

    # Append scores to lists
    em_scores.append(em)
    f1_scores.append(f1)

    # Print results for individual prediction
    print('Question:', test_dict['question'][k])
    print('Answer:', predicted_answer)
    print('Correct Answer:', correct_answer)
    print('Exact Match:', em)
    print('F1 Score:', f1)
    print('---')

# Calculate average and median scores
avg_em = np.mean(em_scores)
median_em = np.median(em_scores)
avg_f1 = np.mean(f1_scores)
median_f1 = np.median(f1_scores)

# Print average and median scores
print('Average EM:', avg_em)
print('Median EM:', median_em)
print('Average F1 Score:', avg_f1)
print('Median F1 Score:', median_f1)



from transformers import pipeline

# Initialize question answering pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# Initialize lists to store EM and F1 scores
em_scores1 = []
f1_scores1 = []

# Iterate over test questions
for i in range(len(test_dict['question'][:20])):
    # Get prediction from question answering model
    result = question_answerer(question=test_dict['question'][i], context=test_dict['context'][i])
    
    # Compute EM and F1 scores for the prediction
    em = compute_exact_match(result['answer'], test_dict['answers'][i]['text'][0])
    f1 = f1_score(result['answer'], test_dict['answers'][i]['text'][0])
    
    # Append scores to lists
    em_scores1.append(em)
    f1_scores1.append(f1)
    
    # Print results for individual question
    print('Question:', test_dict['question'][i])
    print('Answer:', result['answer'])
    print('Correct Answer:', test_dict['answers'][i]['text'][0])
    print('Exact Match:', em)
    print('F1 Score:', f1)
    print('---')

# Calculate average and median scores
avg_em = np.mean(em_scores1)
median_em = np.median(em_scores1)
avg_f1 = np.mean(f1_scores1)
median_f1 = np.median(f1_scores1)

# Print average and median scores
print('Average EM:', avg_em)
print('Median EM:', median_em)
print('Average F1 Score:', avg_f1)
print('Median F1 Score:', median_f1)