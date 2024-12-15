import spacy
from spacy.training.example import Example
import random
import torch
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Enhanced templates
templates = [
    "The {year} {make} {model} with {vehicle_type} body style is available with {odometer} miles on the odometer and a {title_type} title at the {auction_name} auction.",
    "Check out this {color} {year} {make} {model} ({body_style}) with a {transmission} transmission, {fuel_type} fuel, and a {drive_train} drive train. Available near ZIP code {zip_code}.",
    "A {make} {model} ({engine_type}) with {cylinder} cylinders and {odometer} miles, located at {location}, is set for sale on {sale_date}.",
    "Find a {vehicle_type} {year} {make} {model} with a {title_type} title at {auction_name} auction. {fuel_type} engine and {odometer} miles included!",
    "This {color} {make} {model}, a {body_style} from {year}, features a {fuel_type} engine, {drive_train} drive train, and a {transmission} transmission. Sale date: {sale_date}.",
]

# Sample data for attributes (replace with your database values)
car_database = [
    {
        "make": "Ford",
        "model": "Mustang",
        "year": 2018,
        "color": "red",
        "vehicle_type": "Coupe",
        "engine_type": "V8",
        "transmission": "Automatic",
        "fuel_type": "Gasoline",
        "drive_train": "RWD",
        "cylinder": "8",
        "odometer": "45,000",
        "title_type": "Salvage",
        "zip_code": "90210",
        "auction_name": "Copart Los Angeles",
        "location": "Los Angeles, CA",
        "body_style": "Coupe",
        "sale_date": "2024-01-15",
    },
    {
        "make": "Toyota",
        "model": "Camry",
        "year": 2021,
        "color": "blue",
        "vehicle_type": "Sedan",
        "engine_type": "I4",
        "transmission": "Automatic",
        "fuel_type": "Hybrid",
        "drive_train": "FWD",
        "cylinder": "4",
        "odometer": "20,000",
        "title_type": "Clean",
        "zip_code": "94016",
        "auction_name": "IAA San Francisco",
        "location": "San Francisco, CA",
        "body_style": "Sedan",
        "sale_date": "2024-02-01",
    },
    {
        "make": "Chevrolet",
        "model": "Silverado",
        "year": 2020,
        "color": "white",
        "vehicle_type": "Truck",
        "engine_type": "V8",
        "transmission": "Manual",
        "fuel_type": "Diesel",
        "drive_train": "4WD",
        "cylinder": "8",
        "odometer": "60,000",
        "title_type": "Rebuilt",
        "zip_code": "77001",
        "auction_name": "Copart Houston",
        "location": "Houston, TX",
        "body_style": "Pickup",
        "sale_date": "2024-03-10",
    },
    {
        "make": "Tesla",
        "model": "Model 3",
        "year": 2022,
        "color": "black",
        "vehicle_type": "Sedan",
        "engine_type": "Electric",
        "transmission": "Automatic",
        "fuel_type": "Electric",
        "drive_train": "RWD",
        "cylinder": "0",
        "odometer": "10,000",
        "title_type": "Clean",
        "zip_code": "30301",
        "auction_name": "IAA Atlanta",
        "location": "Atlanta, GA",
        "body_style": "Sedan",
        "sale_date": "2024-04-20",
    },
    {
        "make": "Honda",
        "model": "Civic",
        "year": 2019,
        "color": "silver",
        "vehicle_type": "Sedan",
        "engine_type": "I4",
        "transmission": "CVT",
        "fuel_type": "Gasoline",
        "drive_train": "FWD",
        "cylinder": "4",
        "odometer": "30,000",
        "title_type": "Clean",
        "zip_code": "60601",
        "auction_name": "Copart Chicago",
        "location": "Chicago, IL",
        "body_style": "Sedan",
        "sale_date": "2024-05-05",
    },
    {
        "make": "BMW",
        "model": "X5",
        "year": 2017,
        "color": "gray",
        "vehicle_type": "SUV",
        "engine_type": "V6",
        "transmission": "Automatic",
        "fuel_type": "Gasoline",
        "drive_train": "AWD",
        "cylinder": "6",
        "odometer": "50,000",
        "title_type": "Salvage",
        "zip_code": "10001",
        "auction_name": "IAA New York",
        "location": "New York, NY",
        "body_style": "SUV",
        "sale_date": "2024-06-15",
    },
    {
        "make": "Mercedes-Benz",
        "model": "C-Class",
        "year": 2020,
        "color": "white",
        "vehicle_type": "Sedan",
        "engine_type": "I4",
        "transmission": "Automatic",
        "fuel_type": "Gasoline",
        "drive_train": "RWD",
        "cylinder": "4",
        "odometer": "25,000",
        "title_type": "Clean",
        "zip_code": "94102",
        "auction_name": "Copart San Francisco",
        "location": "San Francisco, CA",
        "body_style": "Sedan",
        "sale_date": "2024-07-01",
    },
]

# Generate synthetic training data
training_data = []
for car in car_database:
    for template in templates:
        text = template.format(
            make=car["make"],
            model=car["model"],
            year=car["year"],
            color=car["color"],
            vehicle_type=car["vehicle_type"],
            engine_type=car["engine_type"],
            transmission=car["transmission"],
            fuel_type=car["fuel_type"],
            drive_train=car["drive_train"],
            cylinder=car["cylinder"],
            odometer=car["odometer"],
            title_type=car["title_type"],
            zip_code=car["zip_code"],
            auction_name=car["auction_name"],
            location=car["location"],
            body_style=car["body_style"],
            sale_date=car["sale_date"],
        )
        # Annotate entities
        # Annotate entities
        entities = []
        used_ranges = []

        for key, value in car.items():
            start_idx = text.find(str(value))
            if start_idx != -1:
                end_idx = start_idx + len(str(value))
                overlap = any(start < end_idx and end > start_idx for start, end in used_ranges)
                if not overlap:
                    entities.append((start_idx, end_idx, key.upper()))
                    used_ranges.append((start_idx, end_idx))

        training_data.append({"text": text, "entities": entities})

# Print a sample of the generated data
for example in training_data[:5]:  # Show first 5 examples
    print(example)

# Convert training data for spaCy
spacy_training_data = []
for example in training_data:
    spacy_training_data.append((example["text"], {"entities": example["entities"]}))

# Train spaCy NER model
def train_spacy_ner(training_data, output_dir):
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")

    for _, annotations in training_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    optimizer = nlp.resume_training()
    for i in range(10):
        random.shuffle(training_data)
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses={})

    nlp.to_disk(output_dir)
    print(f"spaCy model saved to {output_dir}")

train_spacy_ner(spacy_training_data, "./spacy_car_model")

# Train BERT NER model
def train_bert_ner(training_data, output_dir):
    # Convert data to BIO format
    def convert_to_bio_format(training_data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bio_data = []
        for example in training_data:
            tokens = tokenizer.tokenize(example["text"])
            labels = ["O"] * len(tokens)
            for start, end, label in example["entities"]:
                entity_text = example["text"][start:end]
                entity_tokens = tokenizer.tokenize(entity_text)
                for i, token in enumerate(entity_tokens):
                    if token in tokens:
                        token_idx = tokens.index(token)
                        labels[token_idx] = f"B-{label}" if i == 0 else f"I-{label}"
            bio_data.append({"tokens": tokens, "ner_tags": labels})
        return bio_data

    bio_data = convert_to_bio_format(training_data)
    train_data, test_data = train_test_split(bio_data, test_size=0.1)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(car_database[0]))

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example['tokens'], truncation=True, padding="max_length", is_split_into_words=True)
        label_ids = [label for label in example['ner_tags']]
        tokenized_inputs['labels'] = label_ids
        return tokenized_inputs

    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"BERT model saved to {output_dir}")

train_bert_ner(training_data, "./bert_car_model")



# Load the saved spaCy model
nlp_spacy = spacy.load("./spacy_car_model")

# Sample text for testing
test_text = "The 2020 Chevrolet Silverado with 60,000 miles is available at the Copart Houston auction."

# Process the text with the spaCy model
doc = nlp_spacy(test_text)

# Print the entities identified by the model
print(f"Entities in the text: {[(ent.text, ent.label_) for ent in doc.ents]}")


# Load the saved BERT model and tokenizer
model = BertForTokenClassification.from_pretrained("./bert_car_model")
tokenizer = BertTokenizer.from_pretrained("./bert_car_model")

# Create a pipeline for token classification
nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)

# Sample text for testing
test_text = "The 2020 Chevrolet Silverado with 60,000 miles is available at the Copart Houston auction."

# Run the model on the test text
entities = nlp_bert(test_text)

# Print the entities identified by the model
print(f"Entities in the text: {entities}")