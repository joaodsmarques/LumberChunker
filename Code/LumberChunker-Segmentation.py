import time
import re
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os


from openai import OpenAI

import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform_v1beta1 import (
    types as aiplatform_types,
    services as aiplatform_services,
)
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
    prediction_service as gapic_prediction_service_types,
    tool as gapic_tool_types,
)
from vertexai.language_models import _language_models as tunable_models

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

#If using ChatGPT
client = OpenAI(api_key="Insert OpenAI key here")

#If using Gemini
project_id = "<put project id here>"
location = "<put project location here>"
vertexai.init(project = project_id, location = location)
model = GenerativeModel('<put model name here>') #by default we use 'gemini-pro'


# Argument parsing
parser = argparse.ArgumentParser(description="Process some text.")
parser.add_argument('--out_path', type=str, default='.', help='Output directory path')
parser.add_argument('--model_type', type=str, required=True, help='ChatGPT (defaults to gpt-3.5-turbo-0125) or Gemini (defaults to gemini-pro)')
parser.add_argument('--book_name', type=str, required=True, help='Example: A_Christmas_Carol_-_Charles_Dickens')



args = parser.parse_args()
out_path = args.out_path
book_name = args.book_name
model_type = args.model_type

# Ensure the output directory exists
create_directory(args.out_path)

if model_type not in ["Gemini", "ChatGPT"]:
    print("Choose Valid Model Type")
    sys.exit(1)


HarmCategory = gapic_content_types.HarmCategory
HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold
FinishReason = gapic_content_types.Candidate.FinishReason
SafetyRating = gapic_content_types.SafetyRating


# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string):
    words = input_string.split()
    return round(1.2*len(words))


# Function to add IDs to each Dataframe Row
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


def LLM_prompt(model_type, user_prompt):
   

    if model_type == "Gemini":
        GenerationConfig = {"temperature": 0.1}
        while True:
            try:
                response = model.generate_content(
                contents = user_prompt,
                generation_config = GenerationConfig,
                #We want to avoid as possible the model refusing the query, hence we set the BlockThresholds to None.
                safety_settings={
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            })

                return response.candidates[0].content.parts[0].text
            except Exception as e:
                # For scenarios where the model still blocks content, we can't redo the prompt, or it will block again. Hence we will increment the ID of the 1st chunk.
                if str(e) == "list index out of range":
                    print("Gemini thinks prompt is unsafe")
                    return "content_flag_increment"
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")
                    time.sleep(60)  # Wait for 1 minute before retrying
    

    elif model_type == "ChatGPT":
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
fileOut  = f'{out_path}/Gemini_Chunks_-_{book_name}.xlsx'



# Filter the DataFrame to show only rows with the specified book name
paragraph_chunks = dataset[dataset['Book Name'] == book_name].reset_index(drop=True)

# Check if the filtered DataFrame is empty
if paragraph_chunks.empty:
    sys.exit("Choose a valid book name!")

id_chunks = paragraph_chunks['Chunk'].to_frame()



# Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
current_id = 0
id_chunks = id_chunks.apply(add_ids, axis=1) # Put ID: Prefix before each paragraph


chunk_number = 0
i = 0


new_id_list = []

word_count_aux = []
current_iteration = 0

while chunk_number < len(id_chunks)-5:
    word_count = 0
    i = 0
    while word_count < 550 and i+chunk_number<len(id_chunks)-1:
        i += 1
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
        word_count = count_words(final_document)
    

    if(i ==1):
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
    else:
        final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i-1 + chunk_number))
    
    
    question = f"\nDocument:\n{final_document}"

    word_count = count_words(final_document)
    word_count_aux.append(word_count)
    chunk_number = chunk_number + i-1

    prompt = system_prompt + question
    gpt_output = LLM_prompt(model_type = model_type, user_prompt=prompt)


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


#Add the last chunk to the list
new_id_list.append(len(id_chunks))

# Remove IDs as they no longer make sense here.
id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)


#Create final dataframe from chunks
new_final_chunks = []
chapter_chunk = []
for i in range(len(new_id_list)):

    # Calculate the start and end indices of each chunk
    start_idx = new_id_list[i-1] if i > 0 else 0
    end_idx = new_id_list[i]
    new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))


    #When building final Dataframe, sometimes text from different chapters is concatenated. When this happens we update the Chapter column accordingly.
    if(paragraph_chunks["Chapter"][start_idx] != paragraph_chunks["Chapter"][end_idx-1]):
        chapter_chunk.append(f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx-1]}")
    else:
        chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])

# Write new Chunks Dataframe
df_new_final_chunks = pd.DataFrame({'Chapter': chapter_chunk, 'Chunk': new_final_chunks})
df_new_final_chunks.to_excel(fileOut, index=False)
print(f"{book_name} Completed!")