# LumberChunker 🪓
This is the official repository for the paper [LumberChunker: Long-Form Narrative Document Segmentation](
https://doi.org/10.48550/arXiv.2406.17526) by André V. Duarte, João D.S. Marques, Miguel Graça, Miguel Freire, Lei Li and Arlindo L. Oliveira<br>

LumberChunker is a method leveraging an LLM to dynamically segment documents into semantically independent chunks. It iteratively prompts the LLM to identify the point within a group of sequential passages where the content begins to shift.

![GitHub Logo](LumberChunker_pipeline.png)


---
## LumberChunker Example - Segmenting a Book
⚠ Important: Whether using Gemini or ChatGPT, don't forget to add the API key / (Project ID, Location) in LumberChunker-Segmentation.py<br>

```
python LumberChunker-Segmentation.py --out_path <output directory path> --model_type <Gemini | ChatGPT> --book_name <target book name>
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
@misc{duarte2024lumberchunker,
      title={LumberChunker: Long-Form Narrative Document Segmentation}, 
      author={André V. Duarte and João Marques and Miguel Graça and Miguel Freire and Lei Li and Arlindo L. Oliveira},
      year={2024},
      eprint={2406.17526},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17526}, 
}
```
