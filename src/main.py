# Standard library imports
import logging
import os.path

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Qdrant imports
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Ragas imports
from ragas.testset import TestsetGenerator, Testset
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer
)
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.engine import RunConfig
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Data processing
import pandas as pd

"""
RAG testset generator main module.
This module provides functionality for generating a testset for a Retrieval-Augmented Generation system.
"""

def load_knowledge_base(directory_path: str) -> list[Document]:
    """
    Load the knowledge base from a directory of documents.
    Load all documents from the specified directory and then split them into chunks using the text_splitter.

    Returns:
        list[Document]: List of chunks of documents loaded from the directory.
    """
    # Load documents from the specified directory using the DirectoryLoader provided by LangChain Community
    loader = DirectoryLoader(directory_path, glob="**/*.pdf")
    # The resulting documents variable is our typical Python list, which holds the chunked versions of the loaded documents, 
    # which can be processed individually in subsequent steps.
    documents = loader.load()
    logger.info(f"{len(documents)} documents loaded.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents_splits = text_splitter.split_documents(documents)
    logger.info(f"{len(documents_splits)} document splits created.")

    return documents_splits

def create_knowledge_graph(documents: list[Document], generator_llm: OllamaLLM, generator_embeddings: OllamaEmbeddings) -> KnowledgeGraph:
    """
    Create a knowledge graph from the loaded documents.

    Returns:
        KnowledgeGraph: The created knowledge graph.
    """
    logger.info("Creating knowledge graph...")
    # Create a KnowledgeGraph using the documents we loaded
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.CHUNK,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
        )

    # For Ragas 0.2.x, we need to convert LangChain models to Ragas models
    try:
        # Wrap LangChain models in Ragas wrappers
        ragas_llm = LangchainLLMWrapper(generator_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)
        # Create the transforms
        transforms = default_transforms(documents=documents, llm=ragas_llm, embedding_model=ragas_embeddings)
        # Apply the transforms to the knowledge graph
        logger.info("Applying transforms to knowledge graph...")
        # Increase timeout and add retry count to handle parsing errors
        run_config = RunConfig(timeout=6000, max_retries=10)  
        apply_transforms(kg, transforms, run_config=run_config)
    except Exception as e:
        logger.error(f"Error applying transforms to knowledge graph: {e}", exc_info=True)
        raise    
    logger.info("Transforms applied to knowledge graph.")
    # Now we have a knowledge graph with additional information, let's save it.
    kg.save("knowledge_graph.json")
    # Load the knowledge graph from the saved file
    loaded_kg = KnowledgeGraph.load("knowledge_graph.json")
    logger.info(f"Knowledge graph built, saved and reloaded. Result:{loaded_kg}")
    return loaded_kg

def setup_models() -> tuple[OllamaLLM, OllamaEmbeddings]:
    """
    Setup the LLM and embedding model for the RAG testset generator.

    Returns:
        tuple[OllamaLLM, OllamaEmbeddings]: Tuple composed of the generator and embedding models.
    """
    logger.info("Setting up models...")
    # Initialize LLMs and embeddings
    # We need three models here:
    # - A generator model that generates the QA pairs based on the provided context.
    generator_llm = OllamaLLM(model="phi3:3.8b")
    # - An embedding to generate embeddings from raw text (will be used to retrieve and generate context).
    ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
    logger.info("Models setup completed.")
    return generator_llm, ollama_emb

def generate_testset(documents: list[Document], generator_llm: OllamaLLM, ollama_emb: OllamaEmbeddings, knowledge_graph: KnowledgeGraph) -> Testset:
    """
    Generate a testset for the RAG system.

    Returns:
        Testset: The generated testset.
    """
    logger.info("Generating testset...")
    # Create Ragas' TestsetGenerator
    generator = TestsetGenerator.from_langchain(llm=generator_llm, embedding_model=ollama_emb, knowledge_graph=knowledge_graph)

    # Specify the distribution of the generated dataset.
    # (Calling default_query_distribution(generator_llm) would produce a distribution equal to the one we specified explicitly.
    # We use explicit set for learning purposes)
    distribution = {
        SingleHopSpecificQuerySynthesizer: 0.5, 
        MultiHopSpecificQuerySynthesizer: 0.25, 
        MultiHopAbstractQuerySynthesizer: 0.25
    }
    # Generate the testset
    testset = generator.generate_with_langchain_docs(documents,
                                                     testset_size=10,
                                                     query_distribution=distribution,
                                                     raise_exceptions=False)
    logger.info(f"Generated testset. Size: {len(testset.to_list())}")
    return testset

def save_testset(testset: Testset, csv_file: str):
    """
    Save the testset to a CSV file.
    """
    # Once the testset has been generated, we can convert it to a Pandas DataFrame
    test_df = testset.to_pandas().dropna()
    logger.info("Testset converted to DataFrame.")
    # Save the testset to CSV
    test_df.to_csv(csv_file, index=False)
    logger.info(f"Testset saved to csv file: {csv_file}")

def create_qdrant_client(qdrant_host: str, qdrant_port: int) -> QdrantClient:
    # Initialize a QdrantClient instance, connecting it to the Qdrant server
    logger.info("Initializing QDRANT client")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Qdrant client initialized successfully at [{qdrant_host}:{qdrant_port}]")
    return client


def main():
    """
    Main entry point for the RAG testset generator.
    """
    logger.info("Starting RAG testset generator initialization...")
    # Get the current script's directory and construct path to docs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Current directory: {current_dir}")
    parent_dir = os.path.dirname(current_dir)
    logger.info(f"Parent directory: {parent_dir}")
    docs_path = os.path.join(parent_dir, "docs")
    logger.info(f"Loading documents from directory: {docs_path}")
    documents_chunks = load_knowledge_base(docs_path)
    generator_llm, ollama_emb = setup_models()
    logger.info("Models initialized.")

    # Create a knowledge graph from the loaded documents
    # We use the KnowledgeGraph to generate a set of scenarios, that are used to generate the testset
    knowledge_graph = create_knowledge_graph(documents_chunks, generator_llm, ollama_emb)
    logger.info("Knowledge graph created.")

    # Generate the testset for the RAG system from the documents composing the knowledge base
    # NOTE: You might not be able to run the above code on your local machine. 
    # If at least a progress bar appears, you know you have done it right. 
    # For the time being, you can utilize a prepared CSV file
    testset = generate_testset(documents_chunks, generator_llm, ollama_emb, knowledge_graph)
    logger.info("Testset generated.")

    # Save the testset
    testset_csv = "generated_testset.csv"
    save_testset(testset, testset_csv)
    logger.info("Testset saved.")

    # Verify everything is correct by loading the testset 
    test_df = pd.read_csv(testset_csv)
    logger.info(f"Testset loaded: {test_df}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    main()
