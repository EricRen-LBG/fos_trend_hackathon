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

# GCP Play with LLMs

Ref: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/text-bison


```python
import numpy as np
import pandas as pd
import json
import time

#from google.colab import auth as google_auth
#google_auth.authenticate_user()

import vertexai
#from vertexai.preview.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel

```

## Project ID and Region

For multiple projects, do:
```
PROJECT_ID = "[your-project-id]"

# Set the project id
! gcloud config set project {PROJECT_ID}
```

```python
PROJECT_ID = ! gcloud config get core/project
PROJECT_ID = PROJECT_ID[0]

REGION = "europe-west2"

PROJECT_ID, REGION
```

<!-- #region toc-hr-collapsed=true -->
## Text Generation
<!-- #endregion -->

### Call from terminal using

```python
%env PROJECT_ID=$PROJECT_ID
```

### Vertex AI SDK for Python

```python

def interview(temperature: float = .2) -> None:
    """Ideation example with a Large Language Model"""

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,
        "max_output_tokens": 256,   
        "top_p": .8,                
        "top_k": 40,                 
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        'Give me ten interview questions for the role of program manager.',
        **parameters,
    )
    print(f"Response from Model: \n{response.text}")
    
```

```python
interview()
```

```python
model = TextGenerationModel.from_pretrained("text-bison@001")
```

```python
prompt = "Who was the first elephant to visit the moon?"
#prompt = "Who was the first elephant to visit the moon? I think it was called Lara"

print(
    model.predict(prompt=prompt, max_output_tokens=256).text
)
```

### Zero shot prompt

```python
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment:
"""

print(model.predict(prompt=prompt, max_output_tokens=256).text)
```

```python

```

## Read the company information  

```python
compnay_info_file = "../fos_complaints_company_2013.csv"
UK_banks = [
    "Barclays Bank UK PLC"
]
```

```python
company_df = pd.read_csv(compnay_info_file, sep=";")
company_df.head()
```

```python
#df[df["company"].str.contains("Bank|Insurance|Barclays|Lloyds|HSBC|Santander|Westminster|Nationwide|Admiral|Aviva", case=False)] # 14633 rows

#df[df["company"].str.contains("Barclays|Lloyds|HSBC|Santander|Westminster|Nationwide|Admiral|Aviva", case=False)] # 5694 rows

company_df = company_df[company_df["company"].str.contains("Lloyds Bank PLC", case=False)]#.sample(5) # 810 rows
company_df.head()
```

## Read the complaint file information

```python
from google.cloud import storage


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    
    return blobs
#    return [blob.name for blob in blobs]
        
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you specify prefix ='a/', without a delimiter, you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a/' and delimiter='/', you'll get back
    only the file directly under 'a/':

        a/1.txt

    As part of the response, you'll also get back a blobs.prefixes entity
    that lists the "subfolders" under `a/`:

        a/b/
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return blobs
#    return [blob.name for blob in blobs]



```

```python
#[b.name for b in list_blobs("fos-trend-bucket")]
[b.name for b in list_blobs_with_prefix(bucket_name="fos-trend-bucket", prefix="text_extracts")]
```

```python
#[b.name for b in list_blobs_with_prefix(bucket_name="fos-trend-bucket", prefix="text_extracts") if b.name in a["drn"] ]

blob_df = pd.DataFrame(
    {
        "blob": [b for b in list_blobs_with_prefix(bucket_name="fos-trend-bucket", prefix="text_extracts")],
        "drn": [b.name[14:25] for b in list_blobs_with_prefix(bucket_name="fos-trend-bucket", prefix="text_extracts")],
    }
)

blob_df.head()
```

```python
selected_df = blob_df.merge(company_df, how="inner", on='drn')

selected_df.head()
```

## Read the complaint text

```python
from io import StringIO

def read_txt(blob):
    #for b in list_blobs_with_prefix(bucket_name="fos-trend-bucket", prefix="text_extracts")]
    f = StringIO(blob.download_as_text(encoding="utf-8"))
    
    return f.read()


#print(read_txt(selected_df["blob"].iloc[0]))
```

```python

```

```python
for index, row in selected_df.iterrows():
    print(row['drn'], row['company'])
    print(read_txt(row["blob"])[0:20])
```

```python
import vertexai
from vertexai.language_models import TextGenerationModel
vertexai.init(project="playpen-714696", location="europe-west4")


def processing_doc(doc_content) -> None:
    """Get one row for the document"""

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "temperature": 0,
        "top_p": 1
    }

#    model = TextGenerationModel.from_pretrained("text-bison@001")
    model = TextGenerationModel.from_pretrained("text-bison")
    
    prompt = f"""You are a text processing agent working with Financial Ombudsman Service (FOS) decision document.
Extract specified values from the source text. 
Return answer as JOSN object with following fields:
 - \"Case number\" <number>
 - \"Complainant\" <string>
 - \"Defendant\" <string>
 - \"Defendant\'s industry\" <string>
 - \"What was the complainant complaining about\" <string>
 - \"What product involved in the complaint\" <string>
 - \"Three key words on this complaint\" <string>
 - \"Three key topics on this complaint\" <string>
 - \"Claimed value in pounds\" <number> 
 - \"When did it happen? (date only)\" <date>
 - \"Final decision (uphold or opposite)\" <string>
 - \"Ombudsman\'s name\" <string>
 - \"Decision deadline\" <date>
 - \"what was the complaint relating to\" <string>
 - \"what was the main reason that the complainant gave for them complaining\"
 - \"What did the complaint want\" <string>
 - \"Summary of what happened\" <string>
 - \"Summary of the Ombudsman\'s reasoning\" <string>


Do not infer any data based on previous training, strictly use only source text given below as input.
========
{doc_content}
========
\"\"\"
"""

    response = model.predict(
        prompt,
        **parameters
    )
   
    return response.text


```

```python
import json

doc_content = read_txt(selected_df["blob"].iloc[0])

response_string = processing_doc({doc_content[0:1000]})

cleaned_json_string = response_string.replace('```','').split("json",1)[1]

d = json.loads(cleaned_json_string)
print(d)

df = pd.json_normalize(d)
df
```

```python



# doc_content = read_txt(selected_df["blob"].iloc[0])

# response_string = processing_doc({doc_content[0:1000]})

# cleaned_json_string = response_string.replace('```','').split("json",1)[1]

# d = json.loads(cleaned_json_string)
# print(d)

# df = pd.json_normalize(d)
# df


#[read_text(r["blob"]) for row in selected_df.iterrows()]

df_list = []
for index, (blob, drn, company) in selected_df.iterrows():
    doc_content = read_txt(blob)
    response_string = processing_doc({doc_content[0:1000]})
    cleaned_json_string = response_string.replace('```','').split("json",1)[1]
    d = json.loads(cleaned_json_string)
    df = pd.json_normalize(d)
    df_list.append(df)

df_list
    
```

```python
summary_table = pd.concat(df_list, axis=0)
```

```python
summary_table_name = "../output/summary_lloyds.csv"

summary_table.to_csv(summary_table_name, index=False)

```

```python

```

```python

```

```python

```

```python

```