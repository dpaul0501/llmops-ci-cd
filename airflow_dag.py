
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import mlflow

# Default args for Airflow
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

dag = DAG('text_processing_pipeline', default_args=default_args, schedule_interval='@daily')

# Tokenize function
def tokenize_data():
    dataset = load_dataset('yelp_review_full')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk('/tmp/tokenized_data')

# Fine-tune LLM
def fine_tune_model():
    dataset = load_from_disk('/tmp/tokenized_data')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    mlflow.set_experiment("LLM_Finetuning_Airflow")
    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test']
        )
        trainer.train()
        mlflow.log_param("epochs", 3)
        mlflow.log_metric("eval_accuracy", trainer.evaluate()["eval_accuracy"])

# Define Airflow tasks
tokenize_task = PythonOperator(
    task_id='tokenize_task',
    python_callable=tokenize_data,
    dag=dag
)

fine_tune_task = PythonOperator(
    task_id='fine_tune_task',
    python_callable=fine_tune_model,
    dag=dag
)

# Set task dependencies
tokenize_task >> fine_tune_task
