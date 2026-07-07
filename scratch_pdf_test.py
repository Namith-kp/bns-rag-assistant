import pypdf

reader = pypdf.PdfReader("data/bns_2023.pdf")
text = reader.pages[27].extract_text() # Somewhere around 103
idx = text.find("Punishment for murder")
if idx != -1:
    snippet = text[idx:idx+30]
    print(repr(snippet))
    for c in snippet:
        print(f"'{c}': {ord(c)}")
