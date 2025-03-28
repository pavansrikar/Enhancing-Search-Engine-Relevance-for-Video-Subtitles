import regex as re
import string
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
db_client = chromadb.PersistentClient(path="searchengine_database")
collection = db_client.get_collection(name="search_engine")

# Load SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'
text_model = SentenceTransformer(model_name, device="cpu")

# Function to preprocess search queries
def clean_text(input_text):
    # Remove line breaks, symbols, and unwanted characters
    patterns = [r"\r\n", r"-->"]
    for pattern in patterns:
        input_text = re.sub(pattern, '', input_text)
    input_text = re.sub("[<>]", "", input_text)
    input_text = re.sub(r'^.*?¶\s*|ï»¿\s*|¶|âª', '', input_text)
    input_text = ''.join([char for char in input_text if char not in string.punctuation and not char.isdigit()])
    return input_text.lower().strip()

# Function to extract numeric IDs from a list
def extract_numeric_ids(ids_list):
    return [match.group(1) for item in ids_list if (match := re.match(r'^(\d+)', item))]

# Streamlit Application Title
st.set_page_config(page_title="Subtitle Search Engine", layout="wide")
st.title("🎥 Enhancing Search Engine Relevance for Video Subtitles")
st.markdown("""
Welcome! This application helps you search for relevant subtitle files by matching your query with subtitle embeddings stored in the database.
""")

# Subtitle search form
with st.form("search_form"):
    st.subheader("🔍 Search for Subtitles")
    search_query = st.text_input("Enter a dialogue to search:", placeholder="Type a line or phrase here...")
    submit_button = st.form_submit_button(label="🔎 Search")

if submit_button:
    if not search_query.strip():
        st.warning("Please enter a valid search query!")
    else:
        # Show loading animation
        with st.spinner("Processing your query..."):
            # Preprocess and encode the user's search query
            cleaned_query = clean_text(search_query)
            query_embedding = model.encode(cleaned_query).tolist()
            
            # Query the subtitle collection
            search_results = collection.query(query_embeddings=query_embedding, n_results=10)
            result_ids = search_results['ids'][0]
            numeric_ids = extract_numeric_ids(result_ids)

        # Display results
        if numeric_ids:
            st.success(f"Found {len(numeric_ids)} matching subtitle files!")
            with st.expander("📂 Relevant Subtitle Files", expanded=True):
                for index, subtitle_id in enumerate(numeric_ids, start=1):
                    file_name = collection.get(ids=f"{subtitle_id}")["documents"][0]
                    st.markdown(f"**{index}.** [{file_name}](https://www.opensubtitles.org/en/subtitles/{subtitle_id})")
        else:
            st.error("No relevant subtitle files found for your query. Please try again with different keywords.")

# Sidebar Information
st.sidebar.header("About this App")
st.sidebar.info("""
This app uses state-of-the-art embedding models to enhance search accuracy for subtitle files. 
Your query is processed and matched using stored subtitle embeddings in the database.
""")

st.sidebar.subheader("Contact")
st.sidebar.write("For support or queries, contact: [support@example.com](mailto:support@example.com)")
