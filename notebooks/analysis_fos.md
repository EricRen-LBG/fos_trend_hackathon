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

# Analysis the FOS summary table for trend analysis



```python
import numpy as np
import pandas as pd
import json
import json5
import os

import time

import openai

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

REGION = "europe-west1"

PROJECT_ID, REGION
```

<!-- #region toc-hr-collapsed=true -->
## For testing the LLM APIs
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

```python

```

## Read the summary table  

```python
summary_table_lloyds = "../output/summary_lloyds.csv"
summary_table_barclays = "../output/summary_barclays.csv"

```

```python
summary_lloyds_df = pd.read_csv(summary_table_lloyds)
summary_barclays_df = pd.read_csv(summary_table_barclays)

summary_df = summary_lloyds_df.append(summary_barclays_df)

print(summary_df.shape)
summary_df.head()

```

```python

```

## Try PandasAI

```python
from pandasai import SmartDataframe

# Sample DataFrame
sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

#openai.api_key  = os.getenv('OPENAI_API_KEY')
my_api_key  = os.getenv('OPENAI_API_KEY')

```

```python

```

```python
# Instantiate a LLM
from pandasai.llm import OpenAI
llm = OpenAI(api_token=my_api_key)

df = SmartDataframe(sales_by_country, config={"llm": llm})
df.chat('Which are the top 5 countries by sales?')
```

### Try some questions

```python
sdf = SmartDataframe(summary_df, config={"llm": llm})

sdf.chat('Which are the top 5 complaints on Lloyds Bank PLC?')
```

```python
summary_df.head()
```

```python
sdf.chat('Which are the top 5 complaints on Lloyds Bank PLC?')
```

```python
sdf.chat('How many complaints against Barclays?')
```

```python
sdf.chat('How many complaints were against Barclays?')
```

```python

```

```python

```

```python

```

## Try Chain of Table?

```python
import sys

sys.path.append("/home/jupyter/dev/fos_trend_hackathon/mix_self_consistency_pack/llama_index/packs/tables") # go to parent dir
sys.path.append("/home/jupyter/mix_self_consistency_pack/llama_index/packs/tables") # go to parent dir

```

```python

```

```python
%env OPENAI_API_KEY=$OPENAI_API_KEY
```

```python
#my_api_key  = os.getenv('OPENAI_API_KEY')

#from llama_index.llms import OpenAI
from llama_index.llms.openai import OpenAI

llm = OpenAI()
#llm = OpenAI(model="gpt-4-1106-preview")
```

```python
#print(my_api_key)
```

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
MixSelfConsistencyPack = download_llama_pack(
    "MixSelfConsistencyPack", 
#    "./mix_self_consistency_pack"
)
```

```python

```

```python
#from mix_self_consistency_pack.base import MixSelfConsistencyQueryEngine
from MixSelfConsistencyPack.base import MixSelfConsistencyQueryEngine


query_engine = MixSelfConsistencyQueryEngine(df=summary_df, llm=llm, verbose=True)

response = query_engine.query(
    "How many complaints against Barclays?"
)
```

```python
query_engine = ChainOfTableQueryEngine(df, llm=llm, verbose=True)

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
ChainOfTablePack = download_llama_pack(
    "ChainOfTablePack", "./chain_of_table_pack"
)
```

```python
!pip install llama-hub
```

```python
import sys
from os.path import dirname
sys.path.append(dirname("/home/jupyter/dev/fos_trend_hackathon/mix_self_consistency_pack/llama_index/packs/tables"))

#"/home/jupyter/dev/fos_trend_hackathon/mix_self_consistency_pack/llama_index/packs/tables"
```

```python
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# documents = SimpleDirectoryReader("../output").load_data()
# index = VectorStoreIndex.from_documents(documents)

```

```python
%pip install llama-index-llms-openai
%pip install llama-hub-llama-packs-tables-chain-of-table-base

```

```python

```
