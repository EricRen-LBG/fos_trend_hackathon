# Google Gen AI Hackathon: FOS Trend Analysis

The problem:

The solution:

## Table Generation using LLMs
Try to use the following prompt to extract key information from FOS documents

    prompt = """
    You are a text processing agent working with Financial Ombudsman Service (FOS) decision document.
    Extract specified values from the source text. 
    Return answer as JOSN object with following fields:
     - "Case number" <number>
     - "Complainant" <string>
     - "Defendant" <string>
     - "Complain on what" <string>
     - "Complaint item" <string>
     - "What product involved in the complaint" <string>
     - "Claimed value in pounds" <number> 
     - "When did it happen? (date only)" <date>
     - "Final decision (uphold or opposite)" <string>
     - "Main reason for the final decision": <string>
     - "Ombudsman's name" <string>
     - "Decision deadline" <date>

    Do not infer any data based on previous training, strictly use only source text given below as input.
    ========
    {fos doc}
    ========
    """

Inference time per doc: ~3s

## Tabular data analysis using LLMs
### Apprach one: PandasAI
 * https://docs.pandas-ai.com/en/latest/

### Approach two: Chain-of-Table
 * Chain-of-Table paper: https://arxiv.org/abs/2401.04398
 * Llama Packs: 
   * https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/tables
   * https://llamahub.ai/l/llama-packs/llama-index-packs-tables?from=
 * Survey paper: https://arxiv.org/pdf/2402.17944v2.pdf
 * Blog: 
   * https://blog.gopenai.com/enhancing-tabular-data-analysis-with-llms-78af1b7a6df9?source=social.linkedin&_nonce=HbvHqvMU
   * https://ameer-hakme.medium.com/unlocking-context-aware-insights-in-tabular-data-with-llms-and-langchain-fac1d33b5c6d
   * https://generative-ai-newsroom.com/can-llms-help-us-understand-data-49891c4e1771
   * https://blog.streamlit.io/chat-with-pandas-dataframes-using-llms/


 
