
import mlflow
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric  # Updated import from the `evaluate` library

# Load dataset and limit to 100 samples
dataset = load_dataset('yelp_review_full', split='train[:100]')

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load accuracy metric
accuracy_metric = load_metric('accuracy')

# Compute accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # Limiting to 1 epoch
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics  # Pass the accuracy computation function
)

# Start fine-tuning and log metrics to MLflow
mlflow.set_experiment("fine-tuning-small")
with mlflow.start_run():
    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics({"accuracy": metrics['eval_accuracy']})  # Logging accuracy metric
    mlflow.log_param("num_samples", 100)

# Save the fine-tuned model
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')  # Ensure tokenizer is saved too
