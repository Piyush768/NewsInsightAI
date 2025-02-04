import streamlit as st
import json

st.sidebar.title("News Article URLs")

# Capture URLs separately
url_1 = st.sidebar.text_input("https://timesofindia.indiatimes.com/entertainment/english/hollywood/news/superman-co-creators-estate-takes-legal-action-against-warner-bros-over-copyright-dispute/articleshow/117858976.cms", "").strip()


# Create a list and filter out empty values
urls = [url for url in [url_1] if url]

st.write("Captured URLs:", urls)

# Ensure at least one URL is entered
if not urls:
    st.warning("Please enter at least one URL before processing.")
    st.stop()

# Convert to JSON and display
json_output = json.dumps(urls, indent=2)
st.write("Formatted JSON Output:", json_output)



