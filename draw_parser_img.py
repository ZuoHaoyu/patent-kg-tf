import spacy
from spacy import displacy
from pathlib import Path

nlp = spacy.load("en_core_web_md")
# doc = nlp("The shaft is positioned through support bearings. A magnetic off-loader provides a magnetic force to move the shaft axially in regard to the bearings.")
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)

print("-------")
doc = nlp("the magnetic force provided levitates the shaft")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

svg = displacy.render(doc, style="dep", jupyter=False)

output_path = Path("images/test.svg")
output_path.open("w", encoding="utf-8").write(svg)