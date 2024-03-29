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

# Processing the FOS documents to generate a summary table for trend analysis

Ref: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/text-bison


```python
import numpy as np
import pandas as pd
import json
import json5

import time

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

```python
BUCKET_NAME = "fos-trend-bucket"
```

<!-- #region toc-hr-collapsed=true -->
## For testing the LLM APIs

**have a laugh first :-)**
<!-- #endregion -->

```python
%env PROJECT_ID=$PROJECT_ID
```

### Vertex AI SDK for Python

```python
model = TextGenerationModel.from_pretrained("text-bison@001")

prompt = "Who was the first elephant to visit the moon?"
#prompt = "Who was the first elephant to visit the moon? I think it was called Lara"

print(
    model.predict(prompt=prompt, max_output_tokens=256).text
)
```

```python

```

## Read the company information  

```python
compnay_info_file = "../fos_complaints_company_2013.csv"
```

```python
company_df = pd.read_csv(compnay_info_file, sep=";")
company_df.head()
```

```python
#df[df["company"].str.contains("Bank|Insurance|Barclays|Lloyds|HSBC|Santander|Westminster|Nationwide|Admiral|Aviva", case=False)] # ~15K rows
#df[df["company"].str.contains("Barclays|Lloyds|HSBC|Santander|Westminster|Nationwide|Admiral|Aviva", case=False)] # ~10K rows

company_df = company_df[company_df["company"].str.contains("Barclays|Lloyds|HSBC|Santander|Westminster|Nationwide|Admiral|Aviva|Insurance", case=False)].sample(1000) # ~10K rows

#company_df = company_df[company_df["company"].str.contains("Lloyds Bank PLC", case=False)].sample(100) # ~1K rows
#company_df = company_df[company_df["company"].str.contains("Barclays", case=False)].sample(100) # 1100 rows

print(company_df.shape)
company_df.head()
```

## Read the complaint file information

```python
from google.cloud import storage


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket.
    
    bucket_name = "your-bucket-name"
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    
    return blobs
        
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

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return blobs

```

```python
[b.name for b in list_blobs_with_prefix(bucket_name=BUCKET_NAME, prefix="text_extracts")][0:10]
```

```python

blob_list = [b for b in list_blobs_with_prefix(bucket_name=BUCKET_NAME, prefix="text_extracts")]
blob_df = pd.DataFrame(
    {
        "blob": blob_list,
        "drn": [b.name[14:25] for b in blob_list],
        "year": [b.name[26:30] for b in blob_list]
    }
)

blob_df.head()
```

```python
selected_df = blob_df.merge(company_df, how="inner", on='drn')

print(selected_df.shape)

selected_df.head()
```

```python

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
for index, row in selected_df.iloc[0:5].iterrows():
    print(row['drn'], row['company'])
    print(read_txt(row["blob"])[0:10])
```

```python
#vertexai.init(project="lloyds-genai24lon-270", location="europe-west1")

def processing_doc(doc_content) -> None:
    """Get one row for the document"""

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "temperature": 0,
        "top_p": 1
    }

#    model = TextGenerationModel.from_pretrained("text-bison")
    model = TextGenerationModel.from_pretrained("text-bison-32k")
    
    prompt = f"""You are a text processing agent working with Financial Ombudsman Service (FOS) decision document.
Extract specified values from the source text. 
Return answer as JSON object with following fields:
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

## Process all

```python
#test one file:
doc_content = read_txt(selected_df["blob"].iloc[1])

response_string = processing_doc({doc_content})
print(response_string)

cleaned_json_string = response_string.replace('```','').replace('"""','').split("json",1)[1]

d = json5.loads(cleaned_json_string)
print(d)

df = pd.json_normalize(d).assign(**{"Original DRN":123, "Company":"XXX"}).set_index("Original DRN", drop=True)

df.head()
```

```python

```

```python
df_list = []

for count, (index, (blob, drn, company)) in enumerate(selected_df.iterrows()):
    print(f"{count}: {drn}")
    try:
        doc_content = read_txt(blob)
        response_string = processing_doc({doc_content})
        # cleaned_json_string = response_string.replace('```','').split("json",1)[1]
        # cleaned_json_string = response_string.replace('```','').replace('"""','')
        cleaned_json_string = response_string.replace('```','').replace('"""','').split("json",1)[1]

        d = json5.loads(cleaned_json_string)
        df = (
            pd.json_normalize(d)
            .assign(**{"Original DRN":drn, "Company":company})
            .set_index("Original DRN", drop=True)
        )
        df_list.append(df)
        print("----------")
    except:
        continue

#df_list[0:2]

summary_table = pd.concat(df_list, axis=0)

```

```python
print(summary_table.shape)
summary_table.head()

summary_table_name = "../output/summary_sampled.csv"

#summary_table_name = "../output/summary_lloyds.csv"
#summary_table_name = "../output/summary_barclays.csv"

summary_table.to_csv(summary_table_name, index=False)
```

```python
selected_df.shape
```

```python

```

```python

```

```python
df = pd.read_csv(summary_table_name).head()

df.shape
```

```python
df.head()
```

```python
! gcloud config get core/project
```

```python
$PROJECT_ID
```

```python
'text_extracts/DRN-1000311_2019-04-15.txt'[26:30]
```

```python

```
