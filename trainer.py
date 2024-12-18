import spacy
from spacy.training.example import Example
import random
import requests
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Solr URL (use your own Solr URL)
solr_url = "https://c-solr9-dev.copart.com/solr/c_dev4_onsale_lots_c/select"

# Solr query parameters
params = {
    "q": "*:*",  # Query all documents
    "q.op": "OR",  # Use OR for query
    "indent": "true",  # Pretty print
    "rows": 1000,  # Limit number of results
}

# Fetch Solr data
response = requests.get(solr_url, params=params)
if response.status_code == 200:
    solr_data = response.json()
    print("Solr data fetched successfully")
else:
    print(f"Error fetching Solr data: {response.status_code}")
    solr_data = None

# Process Solr data
documents = solr_data.get("response", {}).get("docs", []) if solr_data else []

# Enhanced templates
templates = [
    "This {year} {make} {model} with {vehicle_type} body style is available with {odometer} miles and has a {title_type} title. The vehicle is listed with noticeable damage and is available at the {auction_name} auction.",
    "Check out this {color} {year} {make} {model} ({vehicle_type}) with {odometer} miles, a {title_type} title, and visible damage. It's available for bidding at {auction_name}.",
    "A {make} {model} ({year}) with {vehicle_type} body style, {odometer} miles, and a {title_type} title. This vehicle has some damages and is available at the {auction_name} auction.",
    "This {year} {make} {model} with {vehicle_type} body style and {odometer} miles has a {title_type} title. It shows signs of damage and is available for sale at {auction_name}.",
    "Located in {location_city}, {location_state}, this {year} {make} {model} ({vehicle_type}) with {odometer} miles has a {title_type} title and visible damage. It will be auctioned on {sale_date}.",
    "For sale: A damaged {year} {make} {model} ({vehicle_type}), with {odometer} miles and a {title_type} title, available at the {auction_name} auction.",
    "This {year} {make} {model} ({vehicle_type}) with {color} exterior shows signs of damage and has {odometer} miles. Available with a {title_type} title at {auction_name}.",
    "A {make} {model} ({year}) with {vehicle_type} body style and {odometer} miles, this vehicle has been damaged but is still available for sale at the {auction_name} auction with a {title_type} title.",
    "This {year} {make} {model}, featuring a {vehicle_type} body style, is available at the {auction_name} auction with {odometer} miles. The vehicle has visible damage and a {title_type} title.",
    "A {color} {year} {make} {model} with {vehicle_type} body style, {odometer} miles, and a {title_type} title is listed for sale with damages at the {auction_name} auction."
]

# Generate synthetic training data
training_data = []
for car in documents:
    for template in templates:
        try:
            text = template.format(
                make=car.get("lot_make_desc", ""),
                model=car.get("lot_model_desc", ""),
                year=car.get("lot_year", ""),
                color=car.get("lot_color", ""),
                vehicle_type=car.get("body_style", ""),
                cylinder=car.get("cylinders", ""),
                odometer=car.get("odometer_reading_received", ""),
                title_type=car.get("lot_condition_desc", ""),
                auction_name=car.get("yard_name", ""),
                location_city=car.get("location_city", ""),
                location_state=car.get("title_state", ""),
                sale_date=car.get("auction_date_utc", ""),
            )
            print(text)
            # Annotate entities
            entities = []
            used_ranges = []
            for key, value in car.items():
                if not value:
                    continue
                start_idx = text.find(str(value))
                if start_idx != -1:
                    end_idx = start_idx + len(str(value))
                    overlap = any(start < end_idx and end > start_idx for start, end in used_ranges)
                    if not overlap:
                        entities.append((start_idx, end_idx, key.upper()))
                        used_ranges.append((start_idx, end_idx))

            training_data.append({"text": text, "entities": entities})
        except KeyError as e:
            print(f"Missing key in template generation: {e}")

# Convert training data for spaCy
spacy_training_data = [(example["text"], {"entities": example["entities"]}) for example in training_data]

# Train spaCy NER model
def train_spacy_ner(training_data, output_dir):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()
    for i in range(10):
        random.shuffle(training_data)
        losses = {}
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Iteration {i + 1}, Losses: {losses}")

    nlp.to_disk(output_dir)
    print(f"spaCy model saved to {output_dir}")

train_spacy_ner(spacy_training_data, "./spacy_car_model")

# Train BERT NER model
def train_bert_ner(training_data, output_dir):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

    train_data, test_data = train_test_split(bio_data, test_size=0.1)
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)))
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False

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

# train_bert_ner(training_data, "./bert_car_model")

# Test spaCy NER model



# # Test BERT NER model
# model = BertForTokenClassification.from_pretrained("./bert_car_model")
# tokenizer = BertTokenizer.from_pretrained("./bert_car_model")
# nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)
# entities = nlp_bert(test_text)
# print(f"Entities in the text: {entities}")




nlp_spacy = spacy.load("./spacy_car_model")
test_text = "The black 2020 Audi Q4 with 60,000 miles and 8 cylinders is available at the Copart-Houston auction."
doc = nlp_spacy(test_text)
print(f"Entities in the text: {[(ent.text, ent.label_) for ent in doc.ents]}")