from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
import re
import uuid
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.callbacks import get_openai_callback

# Configurație
from pathlib import Path
dotenv_path = Path("/etc/secrets/.env")
load_dotenv(dotenv_path)

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": os.getenv('ALLOWED_ORIGINS', '*')}})
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Memorie conversație
conversation_history = {}
usage_data = defaultdict(list)
EXCEL_FILE = "api_costs.xlsx"

# Costuri
INPUT_COST_PER_TOKEN = 5e-6
OUTPUT_COST_PER_TOKEN = 15e-6

# Inițializare componente
try:
    logger.info("Initializing system...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.6}
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
        model_kwargs={"response_format": {"type": "text"}}
    )

    # Prompturi
    extraction_prompt = PromptTemplate(
        template="""
        [Context]
        {context}
        
        [Task]
        Extrage informații relevante pentru: {question}
        Include detalii complete și termeni tehnici.
        
        [Result]""",
        input_variables=["context", "question"]
    )
    
    synthesis_prompt = PromptTemplate(
        template="""
        [Informații]
        {extracted_info}
        
        [Întrebare]
        {question}
        
        [Cerințe]
        1. Răspuns clar și concis
        2. Ton prietenos
        3. Structurat în puncte când e cazul
        
        [Răspuns]""",
        input_variables=["extracted_info", "question"]
    )

    # Lanțuri
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)
    
    qa_chain = StuffDocumentsChain(
        llm_chain=extraction_chain,
        document_variable_name="context",
        output_key="extracted_info"
    )
    
    logger.info("System ready")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}", exc_info=True)
    raise

# Funcții ajutătoare
def save_to_excel():
    try:
        existing_df = pd.read_excel(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else pd.DataFrame()
        current_df = pd.DataFrame(usage_data)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        combined_df.to_excel(EXCEL_FILE, index=False)
        usage_data.clear()
    except Exception as e:
        logger.error(f"Excel save error: {str(e)}")

def calculate_cost(prompt_tokens, completion_tokens):
    input_cost = prompt_tokens * INPUT_COST_PER_TOKEN
    output_cost = completion_tokens * OUTPUT_COST_PER_TOKEN
    return input_cost, output_cost, input_cost + output_cost

def reformuleaza_intrebare(intrebare):
    replacements = {
        r'\bcat\b': 'cum',
        r'\bunde\b': 'locul unde',
        r'\bcand\b': 'momentul când'
    }
    for pattern, replacement in replacements.items():
        intrebare = re.sub(pattern, replacement, intrebare.lower())
    return intrebare.capitalize()

# Endpoint-uri
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
            
        question = data.get('question', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not question:
            return jsonify({'error': 'Empty question'}), 400

        if session_id not in conversation_history:
            conversation_history[session_id] = deque(maxlen=5)
        
        conversation_history[session_id].append(f"User: {question}")
        
        reformulated = reformuleaza_intrebare(question)
        docs = retriever.get_relevant_documents(reformulated)
        
        with get_openai_callback() as cb:
            extraction_result = qa_chain.run(input_documents=docs, question=reformulated)
            synthesis_result = synthesis_chain.run(
                extracted_info=extraction_result,
                question=reformulated
            )
            
            answer = synthesis_result.replace("[Răspuns]", "").strip()
            answer = re.sub(r'\n+', '\n', answer)
            
            input_cost, output_cost, total_cost = calculate_cost(cb.prompt_tokens, cb.completion_tokens)
            
            usage_data["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            usage_data["session_id"].append(session_id)
            usage_data["question"].append(question)
            usage_data["prompt_tokens"].append(cb.prompt_tokens)
            usage_data["completion_tokens"].append(cb.completion_tokens)
            usage_data["total_cost"].append(total_cost)
            
            if len(usage_data["timestamp"]) >= 5:
                save_to_excel()
        
        conversation_history[session_id].append(f"Assistant: {answer}")
        
        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "usage": {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": total_cost
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "message": str(e)
        }), 500

@app.route('/save_costs', methods=['POST'])
def save_costs():
    save_to_excel()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
