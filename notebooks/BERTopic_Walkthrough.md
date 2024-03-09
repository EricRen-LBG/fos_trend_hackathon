---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (Local)
    language: python
    name: python3
---

<!-- #region id="EOsq6m92LfcF" -->
# BERTopic Walkthrough notebook
<!-- #endregion -->

<!-- #region id="hPiryhCSLp6T" -->
### Set-up
<!-- #endregion -->

```python id="Lzi-CIZ6ICFq"
# !pip install --upgrade --quiet bertopic
# !pip install --upgrade --quiet google-cloud-aiplatform==1.41.0
# !pip install --upgrade --quiet langchain==0.1.6 langchain-google-vertexai==0.0.5
# !pip install --upgrade --quiet PyPDF==4.0.1
# !pip install --upgrade --quiet chromadb==0.4.22
# !pip install --upgrade --quiet ragas==0.1.3
# !pip install --upgrade --quiet tensorflow==2.15
```

```python id="8VSm7J7YLehy"
# Restart kernel after installs so that your environment can access the new packages
import IPython
import time

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<!-- #region id="z28tN4K9LyMs" -->
### Configurations
<!-- #endregion -->

```python id="kZDxK3oyLelB"
import os
PROJECT_ID = ""
# Get Google Cloud project ID from gcloud
if not os.getenv("IS_TESTING"):
    shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
    PROJECT_ID = shell_output[0]
    print("Project ID: ", PROJECT_ID)
```

```python id="Tgwf2J5eLemX"
BUCKET_NAME="playpen-basic-gcp_dv_npd-" + PROJECT_ID + "-bucket"
BUCKET_URL="gs://" + BUCKET_NAME
print("Bucket NAME: ", BUCKET_NAME)
print("Bucket URL: ", BUCKET_URL)
```

```python id="5xJ03vQbLeog"
FILE_BLOB = "rag/fg21-1.pdf"    # Ref.[1]
print("FILE BLOB: ", FILE_BLOB)
```

```python id="5TeWWp2lLer4"
REGION = 'europe-west2'  # London
```

```python id="dDJWurdrLesj"
SERVICE_ACCOUNT = "playpen-5b5a22-consumer-sa@playpen-5b5a22.iam.gserviceaccount.com"  # to be updated per project and service account
```

<!-- #region id="a-p5WVdfL6Xu" -->
### Initialise Vertex AI
<!-- #endregion -->

```python id="xKmCJEEELeup"
import vertexai
vertexai.init(project=PROJECT_ID, location=REGION)
```

<!-- #region id="S2ZuyVDwMEgm" -->
## Scraping
<!-- #endregion -->

```python id="D77on1HiMAdc"
import requests
from bs4 import BeautifulSoup
import regex as re
import pandas as pd
from tqdm import tqdm
```

```python id="MWPlaPu_MAj8"
def get_all_pdf_links(entry_page_url):
    """Extract all pdf links from an url and return a DataFrame with title and pdf url as columns"""

    response = requests.get(url=entry_page_url)
    soup = BeautifulSoup(response.content, "html.parser")

    download_links = soup.find_all(class_="search-result")

    df = pd.DataFrame([
        {"title": pdf_link.find("h4").string, "url": "https://www.financial-ombudsman.org.uk/" + pdf_link.get("href")}
        for pdf_link in download_links
    ])

    return df
```

```python id="uSCNGD4CMA3c"
def get_fos_url(date_from : str  = "2024-01-01" , date_to: str = "2024-01-01", industry_sector_ID: str = "IndustrySectorID%5B1%5D=1"):
    """
    Scrapes text date from (pdf) reports from the FOS Decision website.
    """
    entry_page_url = f"https://www.financial-ombudsman.org.uk/decisions-case-studies/ombudsman-decisions/search?{industry_sector_ID}&DateFrom={date_from}&DateTo={date_to}"
    # Regular expression pattern to match the desired sentence
    pattern = r"Your search returned (\d+) results"

    response = requests.get(url=entry_page_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the matching sentence
    matching_sentence = soup.find(string=re.compile(pattern))

    # Extract the numeric value
    if matching_sentence:
        match = re.search(pattern, matching_sentence)
        result_count = int(match.group(1))
        print(f"Found {result_count} files.")
    else:
        print("No matching sentence found.")
        return None

    total_results_pages = int(result_count/10)+1

    # df_list =[]
    pdf_urls_df = pd.DataFrame()

    for i in tqdm(range(total_results_pages)):
        pdf_urls = entry_page_url+f"&Start={i*10}"
        pdf_urls_df = pd.concat([pdf_urls_df,get_all_pdf_links(pdf_urls)], axis=0, ignore_index=True)

    return pdf_urls_df
```

```python id="9XKGB4SOMKtL"
pdf_url_df = get_fos_url(date_from="2023-12-25", date_to="2024-01-01")

```

```python id="u79FCVcOMKkK"
pdf_url_df.tail()

```

<!-- #region id="Kkg7ANqYMOqU" -->
### Loading PDF Documents
<!-- #endregion -->

```python id="1StMeLj6MBbs"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

```python id="qWQvS87lMBjc"
def check_url_exist(url):
    """To check the url endpoint does exist"""
    response = requests.get(url=url)
    return response.status_code == requests.codes.ok
```

```python id="v0JX0i4sMUWo"
docs = []

for index, row in tqdm(pdf_url_df.iterrows(), total=pdf_url_df.shape[0]):
    if check_url_exist(row.url):

        # -- Loading a pdf file --
        pdf_url = row.url

        loader = PyPDFLoader(pdf_url)
        doc = loader.load()

        splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=1001,
                                            chunk_overlap=250,
                                            separators=["\n\n", "\n", "\. ", " ", ""]
                                        )
        splits = splitter.split_documents(doc)

        docs.extend(splits)
```

```python id="sWQycpoWMUU0"
for doc in docs:
    doc.metadata['file_name'] = doc.metadata['source']
```

```python id="O0F3mSCrMUTv"
# Turn documents into strings, ignoring the metadata
docs_str = []
for doc in docs:
    doc_str = doc.dict()["page_content"]
    docs_str.append(doc_str)
```

<!-- #region id="GJc4s5lMMZ83" -->
## Running initial BERTopic (no tuning)
<!-- #endregion -->

```python id="yCErprk4MUTE"
from bertopic import BERTopic

# Define and fit documens
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs_str)
```

```python id="VqT2LHLrMUPC"
# Show topic information
topic_model.get_topic_info()
```

<!-- #region id="VQkv_jWaNO1M" -->
# BERTopic Full Process
<!-- #endregion -->

```python id="MvxVQR1WMUNx"
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
```

```python id="IWT9M_E1MULj"
# Sentence embedding
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dimensionality Reduction
umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.05,
    metric="cosine",
    random_state=42
)

# Clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# Tokenizer
vectorizer_model = CountVectorizer(
    stop_words = "english",
    # ngram_range=(1,2),
)

# Topic representation
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,
)

# Fine-Tune Representations
keybert_representation = {"keybert": KeyBERTInspired()}
```

<!-- #region id="kbic2AG5N0fS" -->
We can likely play around with different options available here. Notably the parameters in HDBSCAN and the different representations available.

We could also experiment with different sentence embeddings at the start.
<!-- #endregion -->

```python id="slOZK7cRMUJj"
bert_model = BERTopic(
    nr_topics="auto",
    verbose=True,
    vectorizer_model = vectorizer_model,
    ctfidf_model = ctfidf_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    min_topic_size=1,
    representation_model = keybert_representation,
    embedding_model = sentence_model
)
```

```python id="GYveJJIoMUHx"
topics, _ = bert_model.fit_transform(docs_str)
```

```python id="LGh4n0mIMUGo"
topic_labels = bert_model.generate_topic_labels(
    nr_words=5,
    topic_prefix=True,
    word_length=24,
    separator="_",
)
topic_labels
```

```python id="vf5Xf_B4MUD2"
topics_info = bert_model.get_topics()
topics_info
```

```python id="UOIwBF76MUCi"
bert_model.get_topic_info()
```

<!-- #region id="pjMF34E4Pwof" -->
Can we use built-in LLM capability to generate the labels instead?
<!-- #endregion -->

```python id="QfQT3dncMT9l"
from langchain.chains.question_answering import load_qa_chain
from langchain_google_vertexai import VertexAI
from bertopic.representation import LangChain
```

```python id="vpz72lLpMT7R"
chain = load_qa_chain(VertexAI(model_name='gemini-pro', temperature=0.2))
```

```python id="mhyzijmxMT0y"
prompt = "In three words, describe what these documents are about."
representation_model = LangChain(chain, prompt=prompt)
```

```python id="UeS0ElcyQJZm"
# Sentence embedding
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dimensionality Reduction
umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.05,
    metric="cosine",
    random_state=42
)

# Clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# Tokenizer
vectorizer_model = CountVectorizer(
    stop_words = "english",
    # ngram_range=(1,2),
)

# Topic representation
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,
)
```

```python id="95KWQ36EQTF_"
bert_model = BERTopic(
    nr_topics="auto",
    verbose=True,
    vectorizer_model = vectorizer_model,
    ctfidf_model = ctfidf_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    min_topic_size=1,
    representation_model = representation_model,
    embedding_model = sentence_model
)
```

```python id="DqWNyrOOQUPf"
topics, _ = bert_model.fit_transform(docs_str)
```

```python id="eqbR4Oi3QVGs"
topic_labels = bert_model.generate_topic_labels(
    nr_words=5,
    topic_prefix=True,
    word_length=24,
    # separator="_",
)
topic_labels
```

<!-- #region id="rE7wCRPNMBvM" -->
This current set-up is not functioning properly. Can we find a good prompt to help us?
<!-- #endregion -->

<!-- #region id="NgonJalhSlB2" -->
Further extensions:


*   By default each document only contains one topic. We can output the probabilities of each document belonging to a cluster. Can we generalise and improve the outputs.
*   BERTopic has built in dynamic topic modelling. Could this be useful?


<!-- #endregion -->

```python id="j3DLIPPsSh1j"

```
