# LumberChunker 🪵🪓
This is the official repository for the paper LumberChunker: Long-Form Narrative Document Segmentation by André V. Duarte, João Marques, Miguel Graça, Miguel Freire, Lei Li and Arlindo L. Oliveira<br>

LumberChunker is a method leveraging an LLM to dynamically segment documents into semantically independent chunks. It iteratively prompts the LLM to identify the point within a group of sequential passages where the content begins to shift.

![GitHub Logo](LumberChunker_pipeline.png)


---
## LumberChunker Example - Segmenting a Book
⚠ Important: Whether using Gemini or ChatGPT, don't forget to add the API key / (Project ID, Location) in LumberChunker-Segmentation.py<br>

```
python LumberChunker-Segmentation.py --out_path <output directory path> --model_type <Gemini | ChatGPT> --book_name <target book name>
```
Below, we present code example for using ChatGPT to segment a book
```python
import time
import re
import pandas as pd

from openai import OpenAI
client = OpenAI(api_key="insert OpenAI key here")


def count_tokens(input_string):
    words = input_string.split()
    return round(1.2*len(words))


# Function to add IDs
def add_ids(row):
    global current_id
    # Add ID to the chunk
    row['Chunk'] = f'ID {current_id}: {row["Chunk"]}'
    current_id += 1
    return row



system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


def LLM_prompt(user_prompt):
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.1,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ],)
            return completion.choices[0].message.content
        except Exception as e:
            if str(e) == "list index out of range":
                print("GPT thinks prompt is unsafe")
                return "content_flag_increment"
            else:
                print(f"An error occurred: {e}. Retrying in 1 minute...")
                time.sleep(60)  # Wait for 1 minute before retrying

dataset = pd.read_parquet("hf://datasets/LumberChunker/GutenQA_Paragraphs/GutenQA_paragraphs.parquet", engine="pyarrow")
path = "insert path to save files here"

# Insert Book Name Here (We choose A Christmas Carol by Charles_Dickens for the Example)
book_name = "A_Christmas_Carol_-_Charles_Dickens"
fileOut  = f'{path}/Gemini_Chunks_-_{book_name}.xlsx'

# Filter the DataFrame to show only rows with the specified book name
paragraph_chunks = dataset[dataset['Book Name'] == book_name].reset_index(drop=True)
id_chunks = paragraph_chunks['Chunk'].to_frame()


# Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
current_id = 0
id_chunks = id_chunks.apply(add_ids, axis=1) # Put ID: Prefix before each paragraph


chunk_number = 0
new_id_list = []
i = 0

word_count_aux = []
current_iteration = 0

while chunk_number < len(id_chunks)-5:
    word_count = 0
    i = 0
    while word_count < 550 and i+chunk_number<len(id_chunks)-1:
        i += 1
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
        word_count = count_tokens(final_document)
    

    if(i ==1):
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
    else:
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i-1 + chunk_number))
    
    
    question = f"\nDocument:\n{final_document}"

    word_count = count_tokens(final_document)
    word_count_aux.append(word_count)
    chunk_number = chunk_number + i-1

    prompt = system_prompt + question
    gpt_output = LLM_prompt(user_prompt=prompt)


    # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake. This is to avoid being stalled here forever.
    if gpt_output == "content_flag_increment":
        chunk_number = chunk_number + 1

    else:
        pattern = r"Answer: ID \w+"
        match = re.search(pattern, gpt_output)

        if match == None:
            print("repeat this one")
        else:
            gpt_output1 = match.group(0)
            print(gpt_output1)
            pattern = r'\d+'
            match = re.search(pattern, gpt_output1)
            chunk_number = int(match.group())
            new_id_list.append(chunk_number)
            if(new_id_list[-1] == chunk_number):
                chunk_number = chunk_number + 1


new_id_list.append(len(id_chunks))

# Remove IDs as they no longer make sense here.
id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)


#Create final dataframe from chunks
new_final_chunks = []
chapter_chunk = []
for i in range(len(new_id_list)):
    # Calculate the start and end indices for slicing aux1
    start_idx = new_id_list[i-1] if i > 0 else 0
    end_idx = new_id_list[i]
    new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))

    #This is for chunks where Gemini thinks the text should be spanned between 2 different paragraphs
    if(paragraph_chunks["Chapter"][start_idx] != paragraph_chunks["Chapter"][end_idx-1]):
        chapter_chunk.append(f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx-1]}")
    else:
        chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])

# Write new Chunks Dataframe
df_new_final_chunks = pd.DataFrame({'Chapter': chapter_chunk, 'Chunk': new_final_chunks})
df_new_final_chunks.to_excel(fileOut, index=False)
print(f"{book_name} Completed!")
```
---

### 📚 [GutenQA](https://huggingface.co/datasets/LumberChunker/GutenQA)
GutenQA consists of book passages manually extracted from Project Gutenberg and subsequently segmented using LumberChunker.<br>
It features: **100 Public Domain Narrative Books** and **30 Question-Answer Pairs** per Book.<br>

The dataset is organized into the following columns:
- `Book Name`: The title of the book from which the passage is extracted.
- `Book ID`: A unique integer identifier assigned to each book.
- `Chunk ID`: An integer identifier for each chunk of the book. Chunks are listed in the sequence they appear in the book.
- `Chapter`: The name(s) of the chapter(s) from which the chunk is derived. If LumberChunker merged paragraphs from multiple chapters, the names of all relevant chapters are included.
- `Question`: A question pertaining to the specific chunk of text. Note that not every chunk has an associated question, as only 30 questions are generated per book.
- `Answer`: The answer corresponding to the question related to that chunk.
- `Chunk Must Contain`: A specific substring from the chunk indicating where the answer can be found. This ensures that, despite the chunking methodology, the correct chunk includes this particular string.



---
### 📖 GutenQA Alternative Chunking Formats (Used for Baseline Methods)
We also release the same corpus present on GutenQA with different chunk granularities.
- [Paragraph](https://huggingface.co/datasets/LumberChunker/GutenQA_Paragraphs): Books are extracted manually from Project Gutenberg. This is the format of the extraction prior to segmentation with LumberChunker.
- [Recursive Chunks](https://huggingface.co/datasets/LumberChunker/GutenQA_Recursive): Documents are segmented based on a hierarchy of separators such as paragraph breaks, new lines, spaces, and individual characters, using Langchain's [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) function.
- [Semantic Chunks](https://huggingface.co/datasets/LumberChunker/GutenQA_Semantic): Paragraph Chunks are embedded with OpenAI's text-ada-embedding-002. Text is segmented by identifying break points based on significant changes in adjacent chunks embedding distances.
- [Propositions](https://huggingface.co/datasets/LumberChunker/GutenQA_Propositions): Text is segmented as introduced in the paper [Dense X Retrieval](https://arxiv.org/abs/2312.06648). Generated questions are provided along with the correct Proposition Answer.


---
### 🤝 Compatibility
LumberChunker is compatible with any LLM with strong reasoning capabilities.<br>
- In our code, we provide implementation for Gemini and ChatGPT, but in fact models like LLaMA-3, Mixtral 8x7B or Command+R can also be used.<br>


---
## 💬 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{LumberChunker,
      title={{LumberChunker: Long-Form Narrative Document Segmentation}}, 
      author={xxxx, yyyy, zzzz, wwww},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
