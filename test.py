# import torch
# from transformers import BertForTokenClassification, BertTokenizer
# from transformers import pipeline
#
# # Load the saved BERT model and tokenizer
# model = BertForTokenClassification.from_pretrained("./bert_car_model")
# tokenizer = BertTokenizer.from_pretrained("./bert_car_model")
#
# # Create a pipeline for token classification
# nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)
#
# # Sample text for testing
# test_text = "The 2020 Chevrolet Silverado with 60,000 miles is available at the Copart Houston auction."
#
# # Run the model on the test text
# entities = nlp_bert(test_text)
#
# # Print the entities identified by the model
# print(f"Entities in the text: {entities}")


import spacy

# Load the saved spaCy model
nlp_spacy = spacy.load("./spacy_car_model")

# Sample text for testing
test_text = "The 2020 Chevrolet Silverado with 60,000 miles is available at the Copart Houston auction and costs 60000"

# Process the text with the spaCy model
doc = nlp_spacy(test_text)

# Print the entities identified by the model
print(f"Entities in the text: {[(ent.text, ent.label_) for ent in doc.ents]}")

