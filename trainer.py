import os
import spacy
import json
import random
import argparse
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
from spacy.training.example import Example
from datasets import load_dataset


# Function to train spaCy NER model
def train_spacy_ner(training_data, output_dir):
    # Load pre-trained spaCy model
    nlp = spacy.load("en_core_web_trf")  # Transformer-based model

    # Disable unneeded components to speed up training
    nlp.disable_pipes("ner", "parser", "textcat")

    # Add new labels to the NER pipeline
    ner = nlp.get_pipe("ner")
    for label in training_data['labels']:
        ner.add_label(label)

    # Prepare training data
    train_examples = []
    for text, annot in training_data['data']:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        train_examples.append(example)

    # Training loop
    optimizer = nlp.begin_training()
    for i in range(10):  # Epochs
        random.shuffle(train_examples)
        for example in train_examples:
            nlp.update([example], drop=0.5)  # Dropout for regularization

    # Save the fine-tuned model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")


# Function to train BERT model for NER
def train_bert_ner(training_data, output_dir):
    # Load pre-trained BERT model and tokenizer
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(training_data['labels']))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset (in BIO format)
    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
        labels = example["ner_tags"]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Load your custom dataset (replace with your file handling logic)
    dataset = load_dataset('json', data_files=training_data['data'])
    dataset = dataset.map(tokenize_and_align_labels)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    # Start training
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# Main entry point for the service
def main():
    parser = argparse.ArgumentParser(description="Train a custom NER model")
    parser.add_argument("training_data", help="Path to the JSON file containing training data")
    parser.add_argument("output_dir", help="Directory where the fine-tuned model will be saved")
    parser.add_argument("--model", choices=["spacy", "bert"], default="spacy",
                        help="Model type to train (spaCy or BERT)")

    args = parser.parse_args()

    # Load training data
    with open(args.training_data, "r") as f:
        training_data = json.load(f)

    # Train the model
    if args.model == "spacy":
        train_spacy_ner(training_data, args.output_dir)
    elif args.model == "bert":
        train_bert_ner(training_data, args.output_dir)


if __name__ == "__main__":
    main()
