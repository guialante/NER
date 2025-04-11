"""
Custom Named Entity Recognition module for legal documents.
"""
import spacy
import random
from spacy.training import Example
from spacy.util import minibatch, compounding
from pathlib import Path


# Define custom entity labels
CUSTOM_ENTITIES = ["CLIENT", "ASSET", "BENEFICIARY", "LEGAL_CLAUSE"]


def create_training_data():
    """
    Create training data for custom entity recognition.
    
    Returns:
    --------
    list
        A list of tuples (text, entities) where entities is a dict with 'entities' key
        containing a list of (start, end, label) tuples.
    """
    training_data = [
        # CLIENT examples
        ("John Smith has appointed a new attorney.", {
            "entities": [(0, 10, "CLIENT")]
        }),
        ("The client, Mary Johnson, signed the agreement yesterday.", {
            "entities": [(11, 24, "CLIENT")]
        }),
        ("Thomas Anderson is seeking legal advice regarding his estate.", {
            "entities": [(0, 16, "CLIENT")]
        }),
        ("Our firm represents Robert Williams in this matter.", {
            "entities": [(18, 33, "CLIENT")]
        }),
        ("Jane Doe requested a revision to her will.", {
            "entities": [(0, 8, "CLIENT")]
        }),
        # Additional CLIENT examples
        ("Michael Brown consulted with us about his trust.", {
            "entities": [(0, 13, "CLIENT")]
        }),
        ("Patricia Davis wants to update her estate plan.", {
            "entities": [(0, 14, "CLIENT")]
        }),
        ("The testator, William Miller, has three children.", {
            "entities": [(13, 27, "CLIENT")]
        }),
        ("Our client Jennifer Taylor called yesterday.", {
            "entities": [(11, 26, "CLIENT")]
        }),
        ("Richard Moore is the principal in this case.", {
            "entities": [(0, 13, "CLIENT")]
        }),
        
        # ASSET examples
        ("The property at 123 Main Street is included in the trust.", {
            "entities": [(4, 24, "ASSET")]
        }),
        ("His stock portfolio valued at $2.5 million will be transferred.", {
            "entities": [(4, 19, "ASSET")]
        }),
        ("The vacation home in Florida is to be sold.", {
            "entities": [(4, 27, "ASSET")]
        }),
        ("Her retirement accounts should be distributed according to the agreement.", {
            "entities": [(4, 23, "ASSET")]
        }),
        ("The artwork collection will be donated to the museum.", {
            "entities": [(4, 23, "ASSET")]
        }),
        # Additional ASSET examples
        ("The corporate bonds have matured and should be redeemed.", {
            "entities": [(4, 19, "ASSET")]
        }),
        ("His vintage car collection is valued at $350,000.", {
            "entities": [(4, 26, "ASSET")]
        }),
        ("The commercial property in downtown generates rental income.", {
            "entities": [(4, 38, "ASSET")]
        }),
        ("Her jewelry and precious metals are stored in a safe deposit box.", {
            "entities": [(4, 32, "ASSET")]
        }),
        ("The intellectual property rights will transfer to the company.", {
            "entities": [(4, 33, "ASSET")]
        }),
        
        # BENEFICIARY examples
        ("The children will receive equal shares as beneficiaries.", {
            "entities": [(4, 12, "BENEFICIARY")]
        }),
        ("His nephew, Michael Smith, is named as the primary beneficiary.", {
            "entities": [(12, 25, "BENEFICIARY")]
        }),
        ("The charitable organization will receive the remainder.", {
            "entities": [(4, 26, "BENEFICIARY")]
        }),
        ("Sarah Wilson is entitled to 25% of the estate proceeds.", {
            "entities": [(0, 12, "BENEFICIARY")]
        }),
        ("The trust provides for his spouse and minor children.", {
            "entities": [(24, 30, "BENEFICIARY"), (35, 49, "BENEFICIARY")]
        }),
        # Additional BENEFICIARY examples
        ("The university foundation will receive the endowment.", {
            "entities": [(4, 25, "BENEFICIARY")]
        }),
        ("His brother James is named as contingent beneficiary.", {
            "entities": [(4, 16, "BENEFICIARY")]
        }),
        ("The grandchildren will inherit when they reach age 25.", {
            "entities": [(4, 16, "BENEFICIARY")]
        }),
        ("Her caregiver for the last five years receives a special bequest.", {
            "entities": [(4, 13, "BENEFICIARY")]
        }),
        ("The animal shelter is named as a charitable beneficiary.", {
            "entities": [(4, 17, "BENEFICIARY")]
        }),
        
        # LEGAL_CLAUSE examples
        ("According to Article 7, the distribution shall occur within 30 days.", {
            "entities": [(13, 22, "LEGAL_CLAUSE")]
        }),
        ("The heir shall receive the property subject to Section 4.3 of the agreement.", {
            "entities": [(49, 60, "LEGAL_CLAUSE")]
        }),
        ("Paragraph 12 outlines the conditions for early termination.", {
            "entities": [(0, 12, "LEGAL_CLAUSE")]
        }),
        ("The no-contest clause prevents challenges to the will.", {
            "entities": [(4, 22, "LEGAL_CLAUSE")]
        }),
        ("Under the in terrorem provision, any beneficiary who contests the will forfeits their share.", {
            "entities": [(10, 32, "LEGAL_CLAUSE")]
        }),
        # Additional LEGAL_CLAUSE examples
        ("The survivorship clause requires beneficiaries to survive 30 days.", {
            "entities": [(4, 23, "LEGAL_CLAUSE")]
        }),
        ("Per the residuary clause, remaining assets go to charity.", {
            "entities": [(8, 24, "LEGAL_CLAUSE")]
        }),
        ("The powers of appointment section allows the trustee discretion.", {
            "entities": [(4, 29, "LEGAL_CLAUSE")]
        }),
        ("Pursuant to Schedule A, the specific bequests are listed.", {
            "entities": [(13, 23, "LEGAL_CLAUSE")]
        }),
        ("The perpetuities savings clause ensures the trust remains valid.", {
            "entities": [(4, 32, "LEGAL_CLAUSE")]
        }),
        
        # Mixed entity examples
        ("The client James Wilson has designated his daughter Emma as the primary beneficiary of his investment account.", {
            "entities": [(11, 23, "CLIENT"), (53, 57, "BENEFICIARY"), (84, 102, "ASSET")]
        }),
        ("According to Section 3.2, Mary Johnson shall transfer the beach house to her son David.", {
            "entities": [(13, 24, "LEGAL_CLAUSE"), (26, 38, "CLIENT"), (57, 68, "ASSET"), (76, 86, "BENEFICIARY")]
        }),
        ("The trustee shall distribute the bonds to the charitable foundation as specified in Article 9.", {
            "entities": [(19, 29, "ASSET"), (37, 59, "BENEFICIARY"), (78, 87, "LEGAL_CLAUSE")]
        }),
        ("Robert Thompson's collection of antique cars will be sold and the proceeds divided among his three children.", {
            "entities": [(0, 17, "CLIENT"), (31, 43, "ASSET"), (79, 96, "BENEFICIARY")]
        }),
        ("The spendthrift clause protects Sarah Williams' trust assets from creditors.", {
            "entities": [(4, 22, "LEGAL_CLAUSE"), (32, 46, "CLIENT"), (48, 60, "ASSET")]
        }),
        # Additional mixed examples
        ("Pursuant to Article 4, client Elizabeth Brown's retirement accounts shall be divided equally among her four grandchildren.", {
            "entities": [(13, 22, "LEGAL_CLAUSE"), (30, 45, "CLIENT"), (47, 66, "ASSET"), (93, 110, "BENEFICIARY")]
        }),
        ("The executor, Daniel Wilson, shall liquidate the stock portfolio and distribute proceeds to First National Bank as trustee for the minor beneficiaries.", {
            "entities": [(13, 26, "CLIENT"), (48, 63, "ASSET"), (87, 105, "BENEFICIARY"), (119, 139, "BENEFICIARY")]
        }),
        ("The contingent beneficiary clause states that if Jennifer Taylor predeceases the grantor, her share goes to her sister Rebecca.", {
            "entities": [(4, 29, "LEGAL_CLAUSE"), (46, 61, "CLIENT"), (97, 111, "BENEFICIARY")]
        }),
        ("As specified in Schedule B, the vacation property in Aspen shall be held in trust for Thomas Wilson's children until they reach age 30.", {
            "entities": [(16, 26, "LEGAL_CLAUSE"), (32, 55, "ASSET"), (79, 94, "CLIENT"), (96, 104, "BENEFICIARY")]
        }),
    ]
    
    return training_data


def train_custom_ner(output_dir, n_iter=200):
    """
    Train a custom NER model based on spaCy's en_core_web_lg model.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the trained model.
    n_iter : int
        Number of training iterations.
    
    Returns:
    --------
    nlp : spacy.language.Language
        The trained NER model.
    """
    # Load existing spaCy model
    print("Loading base model...")
    nlp = spacy.load("en_core_web_lg")
    
    # Create or get the NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add custom entity labels
    for label in CUSTOM_ENTITIES:
        ner.add_label(label)
    
    # Get the training data
    train_data = create_training_data()
    print(f"Training with {len(train_data)} examples...")
    
    # Disable other pipelines during training for efficiency
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    # Only train NER
    print(f"Beginning training with {n_iter} iterations...")
    with nlp.disable_pipes(*other_pipes):
        # Reset weights for the new labels
        optimizer = nlp.begin_training()
        
        # Batch up the examples
        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Batch training data using spaCy's minibatch and compounding
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            # Update the model for each batch
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                nlp.update(examples, drop=0.5, losses=losses)
            
            if i % 20 == 0:
                print(f"Iteration {i}, Losses: {losses}")
    
    # Save the model
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
    
    return nlp


def extract_custom_entities(nlp, text):
    """
    Extract custom entities from text using a trained spaCy model.
    
    Parameters:
    -----------
    nlp : spacy.language.Language
        The trained NER model.
    text : str
        The input text to extract entities from.
        
    Returns:
    --------
    list of dict
        A list of dictionaries where each dictionary represents an entity.
        Format: {"entity": "John Doe", "type": "CLIENT"}
    """
    # Process the text
    doc = nlp(text)
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        if ent.label_ in CUSTOM_ENTITIES:
            entities.append({
                "entity": ent.text,
                "type": ent.label_
            })
    
    return entities


if __name__ == "__main__":
    # Train the model and save it
    model_dir = "../models/custom_ner"
    train_custom_ner(model_dir, n_iter=200)
    
    # Test the trained model
    test_texts = [
        "Client Thomas Anderson has designated the beach house to his daughter Emma as specified in Article 4.2.",
        "The trustee must distribute Sarah Johnson's retirement accounts to the charitable foundation according to Paragraph 7.",
        "Michael Wilson's estate includes stocks, bonds, and real estate that will be divided among his children and grandchildren.",
        "According to the spendthrift clause, beneficiary Robert Smith cannot sell his interest in the trust."
    ]
    
    # Load the trained model
    print("\nTesting the trained model...")
    trained_nlp = spacy.load(model_dir)
    
    for i, test_text in enumerate(test_texts):
        print(f"\nTest {i+1}: {test_text}")
        entities = extract_custom_entities(trained_nlp, test_text)
        
        if entities:
            print("Extracted custom entities:")
            for entity in entities:
                print(f"  • {entity['entity']} ({entity['type']})")
        else:
            print("No custom entities found.") 