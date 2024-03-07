# Google Gen AI Hackathon: FOS Trend Analysis

The problem:

The solution:

The FOS documents: https://www.financial-ombudsman.org.uk/decisions-case-studies/ombudsman-decisions/search?IndustrySectorID%5B1%5D=1&DateFrom=2023-01-01&DateTo=2023-12-31&IsUpheld%5B1%5D=1&IsUpheld%5B0%5D=0

Each document has around 3000 words or 4000 tokens if we assume 1 token ~ 0.75 words rule. 
Palm2's context windows is 8K

There are ~25k FOS documents each year. In total, we have five year scrapped, around 128k in total

## Table Generation using LLMs
Try to use the following prompt to extract key information from FOS documents

    prompt = """
    You are a text processing agent working with Financial Ombudsman Service (FOS) decision document.
    Extract specified values from the source text. 
    Return answer as JOSN object with following fields:
     - "Case number" <number>
     - "Complainant" <string>
     - "Defendant" <string>
     - "Defendant's industry" <string>
     - "Complain on what" <string>
     - "What product involved in the complaint" <string>
     - "Three key words on this complaint" <string>
     - "Three key topics on this complaint" <string>
     - "Claimed value in pounds" <number> 
     - "When did it happen? (date only)" <date>
     - "Final decision (uphold or opposite)" <string>
     - "Ombudsman's name" <string>
     - "Decision deadline" <date>
     - "What is the complaint about" <string>
     - "What does the complaint want" <string>
     - "Summary of what happened" <string>
     - "Summary of the Ombudsman's reasoning" <string>

    Do not infer any data based on previous training, strictly use only source text given below as input.
    ========
    {fos doc}
    ========
    """


Response from text-bison (latest):  
    """
    {  
      "Case number": "DRN-4107709",  
      "Complainant": "B",  
      "Defendant": "Revolut Ltd",  
      "Defendant's industry": "Financial",  
      "Complain on what": "Unauthorised transactions on his Revolut account",  
      "What product involved in the complaint": "Revolut account",  
      "Three key words on this complaint": "Unauthorised transactions, Fraud, Revolut app",  
      "Three key topics on this complaint": "Unauthorised transactions, Liability for fraudulent transactions, Revolut's investigation",  
      "Claimed value in pounds": 21562,  
      "When did it happen? (date only)": "16 October 2022",  
      "Final decision (uphold or opposite)": "Opposite",  
      "Ombudsman's name": "Dolores Njemanze",  
      "Decision deadline": "28 January 2024",  
      "What is the complaint about": "Revolut's refusal to refund unauthorised transactions on B's account.",  
      "What does the complaint want": "Refund of the unauthorised transactions, interest, and compensation for distress and inconvenience.",  
      "Summary of what happened": "B reported to Revolut and the FOS that his phone and wallet were stolen while he was at a nightclub. He claims that fraudsters accessed his Revolut account and made unauthorised transactions totalling \u00a321,562. Revolut investigated and concluded that the transactions were authorised. B disputed this decision and referred his complaint to the FOS.",  
      "Summary of the Ombudsman's reasoning": "The Ombudsman considered all the evidence and concluded that it was more likely that B made the disputed transactions himself or authorised another party to carry them out. The Ombudsman found that B's testimony was inconsistent and that there were several suspicious factors, such as the delay in reporting his card as lost and the gaps in his memory about the events of the night. The Ombudsman also considered that the transactions were authenticated, either by B's PIN or by his use of the Revolut app, and that Revolut was not required to intervene as the transactions were authorised."  
    }  
    """

Inference time per doc: ~5s

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


 
