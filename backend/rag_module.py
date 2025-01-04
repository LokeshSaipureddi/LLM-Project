import uuid
import logging
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import ChatOllama

class FinancialRAGSystem:
    def __init__(self, db_path="database", pinecone_api_key="", pinecone_index_name="financial-rag"):
        # Initialize logging
        import os
        os.environ['LLAMA_API_KEY'] = ''
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI model
        self.model = ChatOpenAI(temperature=0.5, model="gpt-4o")
        self.llm_model = ChatOllama(
            temperature=0.5, 
            model="Llama-3.2-11B-Vision-Instruct", 
            api_key=os.getenv('LLAMA_API_KEY'),
            base_url="https://models.inference.ai.azure.com"
        )
        # Initialize Hugging Face embeddings
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.hf = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs)

        # Initialize Pinecone
        if not pinecone_api_key:
            import os
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        if not pinecone_api_key:
            raise ValueError("Pinecone API key is required. Pass it as an argument or set PINECONE_API_KEY environment variable.")

        # Initialize Pinecone with a supported region for free plan (gcp-starter)
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create or connect to Pinecone index with a supported configuration
        if pinecone_index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=pinecone_index_name, 
                dimension=768,  # Update to match your embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',  # Use GCP instead of AWS
                    region='us-east-1'  # Use a supported region
                )
            )
        
        self.index = self.pc.Index(host="https://financial-rag-45j6l1s.svc.aped-4627-b74a.pinecone.io")

    def partition_pdf(self, file_path: str, output_path: str) -> tuple:
        """Partitions the PDF into chunks of tables and texts."""
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=1000,
            combine_text_under_n_chars=200,
            new_after_n_chars=600)
        tables = [chunk for chunk in chunks if "Table" in str(type(chunk))]
        texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
        return texts, tables

    def get_images_base64(self, chunks) -> list:
        """Extracts images from CompositeElement objects."""
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64
    
    def text_table_summarizer(self, texts, tables_html):
        # Define the prompt template
        tables = [table.metadata.text_as_html for table in tables_html]
        prompt_text = """
        You are an assistant tasked with summarizing tables, texts, and images.
        Give a concise summary of the table or text or image.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        # Summary chain using OpenAI model
        summarize_chain = {"element": lambda x: x} | prompt | self.model | StrOutputParser()
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})
        return text_summaries, table_summaries
    
    def image_summarizer(self, images):
        prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.model | StrOutputParser()
        image_summaries = chain.batch(images, {"max_concurrency": 2})
        return image_summaries

    def index_documents(self, summaries: dict):
        """Indexes documents into Pinecone."""
        for key, summary_list in summaries.items():
            # Prepare documents for indexing
            for summary in summary_list:
                # Generate a unique ID
                doc_id = str(uuid.uuid4())
                
                # Create embedding
                embedding = self.hf.embed_query(summary)
                
                # Upsert into Pinecone
                self.index.upsert(vectors=[(doc_id, embedding, {"text": summary})])
        
        self.logger.info("Documents indexed successfully in Pinecone")

    def process_and_index(self, file_path: str):
        """Main method to process and index PDF data."""
        self.logger.info(f"Starting to process the PDF file: {file_path}")
            
        # Step 1: Partition PDF into texts and tables
        self.logger.info("Partitioning PDF into texts and tables...")
        texts, tables = self.partition_pdf(file_path, "")
        self.logger.debug(f"Texts extracted: {texts[:3]}")
        self.logger.debug(f"Tables extracted: {tables[:3]}")
        
        # Step 2: Extract images
        self.logger.info("Extracting images and converting to base64...")
        images = self.get_images_base64(texts)
        self.logger.debug(f"Base64 images extracted: {len(images)} images found")
        
        # Step 3: Summarize text and table content
        self.logger.info("Summarizing text and table content...")
        text_summarizes, table_summarizes = self.text_table_summarizer(texts, tables)
        self.logger.debug(f"Text summaries: {text_summarizes[:3]}")
        self.logger.debug(f"Table summaries: {table_summarizes[:3]}")
        
        # Step 4: Summarize image content
        self.logger.info("Summarizing image content...")
        image_summarizes = self.image_summarizer(images)
        self.logger.debug(f"Image summaries: {image_summarizes[:3]}")
        
        # Prepare summaries
        summaries = {
            "texts": text_summarizes,
            "tables": table_summarizes,
            "images": image_summarizes,
        }
        summaries = {key: value for key, value in summaries.items() if value} 
        self.logger.debug(f"Summaries prepared: {summaries}")
        
        # Step 5: Index documents
        if summaries:
            self.index_documents(summaries)
            self.logger.info("Document indexing completed.")
        else:
            self.logger.info("No valid summaries to index. Skipping document indexing.")

    def search(self, text: str, top_k=5):
        """Search documents in Pinecone vector store."""
        # Embed the query
        query_embedding = self.hf.embed_query(text)
        
        # Perform similarity search in Pinecone
        results = self.index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        # Extract and process retrieved documents
        retrieved_docs = []
        for match in results['matches']:
            retrieved_docs.append(match['metadata']['text'])
        
        self.logger.info(f"The meta data is {retrieved_docs}")
        # Construct prompt with retrieved documents
        prompt_template = f"""
        Answer the question based only on the following context:
        Context: {' '.join(retrieved_docs)}
        Question: {text}
        """
        
        # Generate response using the model
        response = self.model.invoke(prompt_template)
        
        return response.content

if __name__=="__main__":
    # Example usage
    f = FinancialRAGSystem()
    print(f.search("Hello"))