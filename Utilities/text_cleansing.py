from bs4 import BeautifulSoup
from spellchecker import SpellChecker
from symspellpy.symspellpy import SymSpell, Verbosity

import re, html, emoji
from contractions import fix
from nltk.corpus import stopwords
import spacy


nlp = spacy.load("en_core_web_sm")

# Step 1: Normalize casing and decode HTML
def normalize_text(text):
    text = text.lower()
    return text

# Option 1.1: Using BeautifulSoup (recommended)
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Option 1.2: Regex (less safe for malformed HTML)
def remove_html_tags_regex(text):
    return re.sub(r'<.*?>', '', text)


# Step 2: Expand contractions
def expand_contractions(text):
    return fix(text)


# Step 3: Remove URLs and mentions
def remove_urls_mentions(text):
    text = re.sub(r"http\S+|www.\S+", " <URL> ", text)
    text = re.sub(r"@\w+", " <USER> ", text)
    return text


# Step 4: Remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def remove_emojis_regex(text):
    # Regular expression pattern to match emoji
    emoji_pattern = re.compile(
                                "[" 
                                "\U0001F600-\U0001F64F"  # Emoticons
                                "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                "\U0001F680-\U0001F6FF"  # Transport & Map symbols
                                "\U0001F700-\U0001F77F"  # Alchemical symbols
                                "\U0001F780-\U0001F7FF"  # Geometric shapes
                                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                "\U00002700-\U000027BF"  # Dingbats
                                "\U000024C2-\U0001F251" 
                                "]+", flags=re.UNICODE
                                )
    return emoji_pattern.sub(r'', text)

# Option 4.1: Counting Emojis
def coount_emojis(text):
    return emoji.emoji_count(text)
    

# Step 5: Handle hashtags (optional: keep content, remove `#`)
def handle_hashtags(text):
    return re.sub(r"#(\w+)", r"\1", text)


# Step 6: Remove special characters (keep basic punctuation)
def remove_special_characters(text):
    # Remove all occurrences of these characters
    pattern = r'[\n\t\r\f\v\b\a\0]'
    text = re.sub(pattern, '', text)
    
    # Remove all occurrences of these characters 2
    pattern = r'[①②③④⑤⑥⑦⑧⑨⑩]'
    text = re.sub(pattern, '', text)
    
    # Remove all occurrences of these characters 3
    pattern = r'[\x00-\x1F]'
    text = re.sub(pattern, '', text)

    return re.sub(r'[^A-Za-z0-9“”\s\".,!?]', '', text)

# Step 7: Normalize whitespace
def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


# Step 8.1: Check/Correct Spelling PYSPELL (SLOWER)
def correct_spelling_pyspell(text):
    spell = SpellChecker()
    corrected_words = []
    for word in text.split():
        if word in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word) or word)
    return ' '.join(corrected_words)

# Step 8.2: Check/Correct Spelling 
def correct_spelling_symspell(text):
    # Create a SymSpell object
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Load dictionary
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
        
    corrected = []
    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(word)
    return ' '.join(corrected)


# Step 9: Correct Grammar
def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    return tool.correct(text)


# Step 10: Lemmatize and optionally remove stopwords
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])


# Final composition functions
def clean_text(text, chk_spelling='', chk_grammar=False, lemmatize=False):
    text = normalize_text(text)
    text = expand_contractions(text)
    text = remove_urls_mentions(text)
    text = remove_emojis(text)
    text = handle_hashtags(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    if chk_spelling == "pyspell":
        text = correct_spelling_symspell(text)
    elif chk_spelling == "symspell":
        text = correct_spelling_symspell(text)

    if chk_grammar == True:
        text = correct_grammar(text)

    if lemmatize == True:
        text = lemmatize_text(text)
        
    return text