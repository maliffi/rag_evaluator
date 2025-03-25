# RAG Testset Generator

A Python project for generating test datasets to evaluate Retrieval-Augmented Generation (RAG) systems. This tool creates diverse query-answer pairs from your documents to help you assess the performance of your RAG implementations.

## Project Domain

This project addresses a critical challenge in developing and evaluating RAG systems: creating comprehensive test datasets that measure how well your RAG system retrieves relevant information and generates accurate responses. It uses:

- **Knowledge Graph Generation**: Creates a semantic representation of your documents
- **Query Synthesis**: Generates various types of queries (single-hop, multi-hop specific, multi-hop abstract)
- **Test Dataset Creation**: Produces structured test cases with questions, contexts, and expected answers

## Project Structure

- `src/` - Source code
  - `main.py` - Main entry point with the core functionality
- `docs/` - Directory where you should place your PDF documents to be processed
- `requirements.txt` - Project dependencies
- `knowledge_graph.json` - Generated knowledge graph for the document set
- `generated_testset.csv` - Output file containing the generated test dataset

## Process Implementation

The main process implemented in `main.py` follows these steps:

1. **Document Loading**: Loads PDF documents from the `docs/` directory
2. **Document Chunking**: Splits documents into manageable chunks for processing
3. **Model Setup**: Initializes LLM (Ollama Phi3:3.8b) and embedding model (Nomic Embed Text) 
4. **Knowledge Graph Creation**: 
   - Creates a graph representation of your documents
   - Applies transforms to enhance the graph with additional information
5. **Test Set Generation**:
   - Generates test cases with different query types:
     - 50% Single-hop specific queries (focusing on specific information)
     - 25% Multi-hop specific queries (requiring combining information from multiple sources)
     - 25% Multi-hop abstract queries (requiring higher-level reasoning)
6. **Test Set Storage**: Saves the generated test set to CSV for further use

## Running Locally

### Setup

1. Clone this repository
2. Activate a virtual environment:

```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Generator

1. **Prepare Your Documents**:
   - Place your PDF documents in the `docs/` directory at the root of the project
   - The system will process all PDFs in this directory (including subdirectories)

2. **Run the Generator**:

```bash
python src/main.py
```

3. **Access Results**:
   - The generated test dataset will be saved as `generated_testset.csv`
   - The knowledge graph will be saved as `knowledge_graph.json`

### Requirements

- **Ollama**: Must be installed and running locally with the following models:
  - `phi3:3.8b` for generation
  - `nomic-embed-text` for embeddings

## Dependencies

This project relies on several key libraries:
- LangChain for document processing and orchestration
- RAGAS for test set generation
- Ollama for local LLM access
- Qdrant for vector storage
- Unstructured for PDF processing

See `requirements.txt` for the complete list of dependencies.
