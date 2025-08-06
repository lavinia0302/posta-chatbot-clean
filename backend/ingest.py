import os
import sys
import logging
import json
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Actualizat la noua importare
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def safe_file_operation(filepath, func):
    """Wrapper pentru operații sigure cu fișiere"""
    try:
        # Normalizare cale pentru Windows și gestionare căi lungi
        filepath = os.path.normpath(filepath)
        if not os.path.exists(filepath):
            # Verificare alternativă pentru căi lungi
            if len(filepath) > 260:
                try:
                    filepath = '\\\\?\\' + os.path.abspath(filepath)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError
                except:
                    logger.warning(f"Cale prea lungă sau fișier negăsit: {filepath[:100]}...")
                    return None
            else:
                logger.warning(f"Fișierul nu există: {filepath}")
                return None
        return func(filepath)
    except Exception as e:
        logger.warning(f"Eroare la procesarea {os.path.basename(filepath)}: {str(e)}")
        return None

def get_file_hash(filepath):
    """Generează hash MD5 pentru un fișier"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Eroare la citirea {filepath}: {str(e)}")
        return None

def process_single_file(filepath):
    """Procesează un singur fișier"""
    def _process(filepath):
        try:
            loader = TextLoader(filepath, encoding='utf-8', autodetect_encoding=True)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata.update({
                    'source_file': os.path.basename(filepath),
                    'file_size': os.path.getsize(filepath),
                    'processing_date': datetime.now().isoformat()
                })
            return docs
        except Exception as e:
            logger.error(f"Eroare la încărcarea {os.path.basename(filepath)}: {str(e)}")
            return []

    return safe_file_operation(filepath, _process)

def validate_documents(documents):
    """Verifică documentele pentru conținut valid"""
    if not documents:
        logger.error("Niciun document valid pentru procesare")
        return []
        
    valid_docs = [doc for doc in documents if doc and doc.page_content.strip()]
    logger.info(f"Documente valide găsite: {len(valid_docs)}")
    return valid_docs

def load_and_process_documents():
    """Încarcă și procesează documentele"""
    logger.info("Pornire proces ingest")
    
    # Verificare director
    if not os.path.exists("posta_romana"):
        logger.error("Directorul 'posta_romana' nu există!")
        return False

    # Detectare fișiere
    all_files = []
    for root, _, files in os.walk("posta_romana"):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                all_files.append(filepath)

    if not all_files:
        logger.error("Niciun fișier .txt găsit în directorul posta_romana")
        return False

    logger.info(f"Fișiere .txt identificate: {len(all_files)}")

    # Procesare paralelă
    documents = []
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, all_files),
            total=len(all_files),
            desc="Procesare fișiere"
        ))
        documents = [doc for doc in results if doc is not None]

    # Aplatizare lista de documente
    documents = [doc for sublist in documents for doc in sublist]
    
    # Validare documente
    valid_documents = validate_documents(documents)
    if not valid_documents:
        logger.error("Niciun document valid după procesare")
        return False

    # Creare vectorstore
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splitted_docs = splitter.split_documents(valid_documents)
        logger.info(f"Chunk-uri create: {len(splitted_docs)}")
        
        # Configurare embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Creare vectorstore cu progress bar alternativ
        logger.info("Creare vectorstore...")
        with tqdm(total=len(splitted_docs), desc="Procesare chunk-uri") as pbar:
            vectorstore = FAISS.from_documents(
                splitted_docs, 
                embeddings
            )
            pbar.update(len(splitted_docs))
        
        # Salvare
        vectorstore.save_local("vectorstore")
        logger.info(f"Vectorstore creat cu {len(splitted_docs)} chunk-uri")
        return True
        
    except Exception as e:
        logger.error(f"Eroare la crearea vectorstore: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        success = load_and_process_documents()
        if success:
            logger.info("Proces finalizat cu succes")
            exit(0)
        else:
            logger.error("Proces terminat cu erori")
            exit(1)
    except Exception as e:
        logger.error(f"Eroare critică: {str(e)}", exc_info=True)
        exit(1)