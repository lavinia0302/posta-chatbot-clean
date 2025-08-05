import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
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
from langchain.callbacks import get_openai_callback

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')

# Memorie conversațională per sesiune
conversation_history = {}

# Structură pentru stocarea costurilor
usage_data = defaultdict(list)
EXCEL_FILE = "/tmp/api_costs.xlsx"  # Folosim /tmp pentru compatibilitate cu Render

# Prețuri pe token
INPUT_COST_PER_TOKEN = 5e-6
OUTPUT_COST_PER_TOKEN = 15e-6

@app.before_request
def init_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if session['session_id'] not in conversation_history:
        conversation_history[session['session_id']] = deque(maxlen=5)

def get_context():
    return "\n".join(conversation_history[session['session_id']])

def save_to_excel():
    try:
        if os.path.exists(EXCEL_FILE):
            existing_df = pd.read_excel(EXCEL_FILE)
        else:
            existing_df = pd.DataFrame()

        current_df = pd.DataFrame(usage_data)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        combined_df.to_excel(EXCEL_FILE, index=False)
        usage_data.clear()
    except Exception as e:
        logger.error(f"Eroare la salvarea datelor: {str(e)}")

def calculate_cost(prompt_tokens, completion_tokens):
    return (
        prompt_tokens * INPUT_COST_PER_TOKEN,
        completion_tokens * OUTPUT_COST_PER_TOKEN,
        (prompt_tokens * INPUT_COST_PER_TOKEN) + (completion_tokens * OUTPUT_COST_PER_TOKEN)
    )

# Inițializare componente
try:
    logger.info("Încărcare sistem...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Încărcare vectorstore cu cale relativă
    vectorstore = FAISS.load_local(
        "./vectorstore", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
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
    
    logger.info("Sistem gata")

except Exception as e:
    logger.error(f"Eroare inițializare: {str(e)}", exc_info=True)
    raise

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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
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
            
            # Salvarea datelor de utilizare
            usage_data["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            usage_data.update({
                "session_id": session_id,
                "question": question,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": (cb.prompt_tokens * INPUT_COST_PER_TOKEN) + (cb.completion_tokens * OUTPUT_COST_PER_TOKEN)
            })
            
            if len(usage_data["timestamp"]) >= 5:
                save_to_excel()
            
            return jsonify({
                "answer": answer,
                "usage": {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": (cb.prompt_tokens * INPUT_COST_PER_TOKEN) + (cb.completion_tokens * OUTPUT_COST_PER_TOKEN)
                }
            })
            
    except Exception as e:
        logger.error(f"Eroare: {str(e)}")
        return jsonify({"error": "Eroare la procesare"}), 500

@app.route('/save_costs', methods=['POST'])
def save_costs():
    try:
        save_to_excel()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)