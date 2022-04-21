import tensorflow as tf
import sys
import json
from spacy.lang.en import English
import pandas as pd

print(tf.__version__)
print(sys.version)

# with open("Notes/skimlit_example_abstracts.json", "r") as f:
#     example_abstracts = json.load(f)
#
# abstracts = pd.DataFrame(example_abstracts)
#
# nlp = English()
# nlp.add_pipe('sentencizer')
#
# doc = nlp(example_abstracts[0]["abstract"])  # create "doc" of parsed sequences, change index for a different abstract
# abstract_lines = [str(sent) for sent in list(doc.sents)]  # return detected sentences from doc in string type (not spaCy token type)
#
# for line in abstract_lines:
#     print(line)
