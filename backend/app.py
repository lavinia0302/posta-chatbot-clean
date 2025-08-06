import os
import logging
import re
import uuid
import tempfile
from datetime import datetime
from collections import deque, defaultdict
from functools import wraps

import pandas as pd
import psutil
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks import get_openai_callback

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/chat": {"origins": os.getenv("ALLOWED_ORIGINS", "*")},
    r"/metrics": {"origins": ""}
})

# Configurare cheie secretă
app.secret_key = os.environ['FLASK_SECRET_KEY']

# Constante
INPUT_COST_PER_TOKEN = 5e-6
OUTPUT_COST_PER_TOKEN = 15e-6
EXCEL_FILE = os.path.join(tempfile.gettempdir(), "api_costs.xlsx")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 10))

# Memorie conversațională și date de utilizare
conversation_history = {}
usage_data = defaultdict(list)

# Middleware pentru autentificare metrici
def require_api_key(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if request.headers.get('X-API-KEY') != os.getenv('METRICS_API_KEY'):
            logger.warning("Acces neautorizat la endpoint-ul de metrici")
            return jsonify({"error": "Unauthorized"}), 401
        return view_func(*args, **kwargs)
    return wrapped

# Inițializare componente NLP
try:
    logger.info("Încărcare sistem NLP...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(
        os.getenv("VECTORSTORE_PATH", "./vectorstore"),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": 0.6}
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
        model_kwargs={"response_format": {"type": "text"}}
    )
    
    extraction_prompt = PromptTemplate(
        template="""
        [Context] {context}
        [Sarcină] Extrage informații relevante pentru: {question}
        [Rezultat]""",
        input_variables=["context", "question"]
    )
    
    synthesis_prompt = PromptTemplate(
        template="""
        [Informații] {extracted_info}
        [Întrebare] {question}
        [Răspuns]""",
        input_variables=["extracted_info", "question"]
    )
    
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)
    
    qa_chain = StuffDocumentsChain(
        llm_chain=extraction_chain,
        document_variable_name="context",
        output_key="extracted_info"
    )
    
    logger.info("Sistem NLP încărcat cu succes")

except Exception as e:
    logger.critical(f"Eroare inițializare sistem: {str(e)}", exc_info=True)
    raise

# Utilitare
def init_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if session['session_id'] not in conversation_history:
        conversation_history[session['session_id']] = deque(maxlen=5)

def save_to_excel():
    try:
        df = pd.DataFrame(usage_data)
        df.to_excel(EXCEL_FILE, index=False, mode='a' if os.path.exists(EXCEL_FILE) else 'w')
        usage_data.clear()
    except Exception as e:
        logger.error(f"Eroare salvare Excel: {str(e)}", exc_info=True)

def reformuleaza_intrebare(intrebare):
    replacements = {
        r'\bcat\b': 'cum',
        r'\bunde\b': 'locul unde',
        r'\bcand\b': 'momentul când',
        r'\bde ce\b': 'motivul pentru care'
    }
    for pattern, replacement in replacements.items():
        intrebare = re.sub(pattern, replacement, intrebare.lower())
    return intrebare.capitalize()

# Endpoint-uri
@app.before_request
def before_request():
    init_session()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            logger.warning("Întrebare goală primită")
            return jsonify({'error': 'Întrebare goală'}), 400

        session_id = session['session_id']
        conversation_history[session_id].append(f"Utilizator: {question}")
        
        with get_openai_callback() as cb:
            docs = retriever.get_relevant_documents(reformuleaza_intrebare(question))
            extraction_result = qa_chain.run(input_documents=docs, question=question)
            answer = synthesis_chain.run(
                extracted_info=extraction_result,
                question=question
            ).replace("[Răspuns]", "").strip()
            
            conversation_history[session_id].append(f"Asistent: {answer}")
            
            # Logare utilizare
            usage_data["timestamp"].append(datetime.now().isoformat())
            usage_data.update({
                "session_id": session_id,
                "question": question,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": (cb.prompt_tokens * INPUT_COST_PER_TOKEN) + 
                             (cb.completion_tokens * OUTPUT_COST_PER_TOKEN)
            })
            
            if len(usage_data["timestamp"]) >= 5:
                save_to_excel()
            
            return jsonify({
                "answer": answer,
                "usage": {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            })
            
    except Exception as e:
        logger.error(f"Eroare procesare chat: {str(e)}", exc_info=True)
        return jsonify({"error": "Eroare la procesare"}), 500

@app.route('/metrics')
@require_api_key
def metrics():
    return jsonify({
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "active_sessions": len(conversation_history)
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "components": {
            "vectorstore": "active",
            "llm": "active"
        }
    }), 200

@app.route('/save_costs', methods=['POST'])
def save_costs():
    try:
        save_to_excel()
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Eroare salvare costuri: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'false').lower() == 'true')

