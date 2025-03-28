import pandas as pd
import sqlite3
import zipfile
import io
from tqdm import tqdm
import re
import nltk
import string
from sentence_transformers import SentenceTransformer
import chromadb

# Database connection
db_path = r""C:\Users\srikar\Downloads\eng_subtitles_database.db"
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# Fetch and display table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())

# Load data into DataFrame
data = pd.read_sql_query("SELECT * FROM zipfiles", connection)
print(data.head())

# Initialize progress bar
tqdm.pandas()

# Decode zip file content
def extract_zip_content(byte_data):
    with zipfile.ZipFile(io.BytesIO(byte_data)) as zf:
        first_file = zf.namelist()[0]
        content = zf.read(first_file)
    return content.decode('latin-1')

data['content'] = data['content'].progress_apply(extract_zip_content)

# Data cleaning
nltk.download('stopwords')

def clean_text(doc):
    doc = re.sub(r"\r\n", '', doc)
    doc = re.sub(r"-->", "", doc)
    doc = re.sub("[<>]", "", doc)
    doc = re.sub(r'^.*?¶\s*|ï»¿\s*|¶|âª', '', doc)
    doc = ''.join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    return doc.lower().strip()

data['content'] = data['content'].progress_apply(clean_text)

# Sampling
sampled_data = data.sample(frac=0.3, random_state=42)

# Chunk text into smaller segments
def create_text_chunks(text, size, overlap):
    segments = []
    start_idx = 0
    while start_idx < len(text):
        chunk = text[start_idx:start_idx + size]
        segments.append(chunk.lower())
        start_idx += size - overlap
    return segments

chunk_size = 500
overlap_size = 100
sampled_data['chunks'] = sampled_data['content'].progress_apply(
    lambda x: create_text_chunks(x, chunk_size, overlap_size)
)

# Sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
sampled_half = sampled_data.sample(frac=0.5, random_state=42)
sampled_half['encoded_data'] = sampled_half['chunks'].apply(
    lambda chunks: model.encode(chunks).tolist()
)

# Initialize ChromaDB
db_client = chromadb.PersistentClient(path="searchengine_database")
collection = db_client.get_or_create_collection(name="search_engine", metadata={"hnsw:space": "cosine"})

def store_embeddings_to_db(df):
    for index, row in df.iterrows():
        collection.add(
            documents=[row['name']],
            embeddings=row['encoded_data'],
            ids=[str(row['num'])]
        )

store_embeddings_to_db(sampled_half)
