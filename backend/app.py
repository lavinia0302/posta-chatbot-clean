from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import re
import uuid
import time
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
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-here'

# Memorie conversațională per sesiune
conversation_history = {}

# Structură pentru stocarea costurilor
usage_data = defaultdict(list)
EXCEL_FILE = "api_costs.xlsx"

# Prețuri pe token (în dolari)
INPUT_COST_PER_TOKEN = 5e-6  # $5 per 1M tokens
OUTPUT_COST_PER_TOKEN = 15e-6  # $15 per 1M tokens

@app.before_request
def init_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if session['session_id'] not in conversation_history:
        conversation_history[session['session_id']] = deque(maxlen=5)

def get_context():
    """Returnează contextul conversației"""
    return "\n".join(conversation_history[session['session_id']])

def save_to_excel():
    """Salvează datele de utilizare în fișierul Excel"""
    try:
        # Încarcă datele existente dacă fișierul există
        try:
            existing_df = pd.read_excel(EXCEL_FILE)
        except FileNotFoundError:
            existing_df = pd.DataFrame()

        # Creează un DataFrame din datele curente
        current_df = pd.DataFrame(usage_data)
        
        # Combină cu datele existente
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
        
        # Salvează în fișier
        combined_df.to_excel(EXCEL_FILE, index=False)
        logger.info(f"Datele au fost salvate în {EXCEL_FILE}")
        
        # Golește datele din memorie după salvare
        usage_data.clear()
    except Exception as e:
        logger.error(f"Eroare la salvarea în Excel: {str(e)}")

def calculate_cost(prompt_tokens, completion_tokens):
    """Calculează costul pe baza numărului de tokens"""
    input_cost = prompt_tokens * INPUT_COST_PER_TOKEN
    output_cost = completion_tokens * OUTPUT_COST_PER_TOKEN
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

# Inițializare componente (păstrăm aceeași configurare ca înainte)
try:
    logger.info("Încărcare sistem...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.6
        }
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
        model_kwargs={"response_format": {"type": "text"}}
    )
    
    # Prompt pentru extragere informații
    extraction_prompt = PromptTemplate(
        template="""
        [Context]
        {context}
        
        [Sarcină]
        Extrage toate informațiile relevante pentru: {question}
        Include toate detaliile, condițiile și excepțiile găsite.
        
        [Format]
        - Listă completă a tuturor punctelor relevante
        - Menționează dacă există informații incomplete
        - Păstrează termenii tehnici originali
        
        [Rezultat]
        """,
        input_variables=["context", "question"]
    )
    
    # Prompt pentru sinteză
    synthesis_prompt = PromptTemplate(
        template="""
        [Informații găsite]
        {extracted_info}
        
        [Întrebare]
        {question}
        
        [Sarcină]
        1. Compune un răspuns clar și concis bazat pe context
        2. Păstrează un ton prietenos și conversațional
        3. Structurază răspunsul în puncte clare când este relevant
        4. Folosește informații din contextul conversației când este cazul
        
        [Restricții]
        - Răspunsul trebuie să fie relevant pentru întrebare
        - Menționează dacă nu ai suficiente informați
        
        [Răspuns]
        """,
        input_variables=["extracted_info", "question"]
    )
    
    # Creare lanțuri
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
    """Simplifică întrebările pentru a îmbunătăți înțelegerea"""
    intrebare = intrebare.lower()
    replacements = {
        r'\bcat\b': 'cum',
        r'\bunde\b': 'locul unde',
        r'\bcand\b': 'momentul când',
        r'\bde ce\b': 'motivul pentru care'
    }
    
    for pattern, replacement in replacements.items():
        intrebare = re.sub(pattern, replacement, intrebare)
    
    return intrebare.capitalize()

def generate_answer(question, docs):
    """Generează răspuns conversațional și returnează costul"""
    try:
        context = get_context()
        full_question = f"{context}\n\nÎntrebare: {question}" if context else question
        
        with get_openai_callback() as cb:
            extraction_result = qa_chain.run(input_documents=docs, question=full_question)
            synthesis_result = synthesis_chain.run(
                extracted_info=extraction_result,
                question=full_question
            )
            
            # Calculăm costul
            input_cost, output_cost, total_cost = calculate_cost(cb.prompt_tokens, cb.completion_tokens)
            
            # Post-procesare pentru a face răspunsul mai natural
            synthesis_result = synthesis_result.replace("[Răspuns]", "").strip()
            synthesis_result = re.sub(r'\n+', '\n', synthesis_result)
            
            return {
                "answer": synthesis_result,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
    except Exception as e:
        logger.error(f"Eroare generare răspuns: {str(e)}")
        return {
            "answer": "Nu am putut procesa întrebarea. Poți reformula, te rog?",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0
        }

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Întrebare goală.'}), 400

        # Adăugăm întrebarea la istoric
        conversation_history[session['session_id']].append(f"Utilizator: {question}")
        
        # Reformulare pentru a îmbunătăți înțelegerea
        reformulated = reformuleaza_intrebare(question)
        
        # Obține documentele relevante
        docs = retriever.get_relevant_documents(reformulated)
        
        # Generează răspunsul și obține costul
        result = generate_answer(reformulated, docs)
        answer = result["answer"]
        
        # Adăugăm răspunsul la istoric
        conversation_history[session['session_id']].append(f"Asistent: {answer}")
        
        # Formatare finală
        answer = answer.replace("• ", "- ").strip()
        
        # Salvează datele de utilizare
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        usage_data["timestamp"].append(timestamp)
        usage_data["session_id"].append(session['session_id'])
        usage_data["question"].append(question)
        usage_data["prompt_tokens"].append(result["prompt_tokens"])
        usage_data["completion_tokens"].append(result["completion_tokens"])
        usage_data["input_cost"].append(result["input_cost"])
        usage_data["output_cost"].append(result["output_cost"])
        usage_data["total_cost"].append(result["total_cost"])
        
        # Salvează periodic în Excel (de exemplu, la fiecare 5 cereri)
        if len(usage_data["timestamp"]) >= 5:
            save_to_excel()
        
        return jsonify({
            "answer": answer,
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_cost": result["total_cost"]
            }
        })

    except Exception as e:
        logger.error(f"Eroare în endpoint /chat: {str(e)}")
        return jsonify({"error": "Eroare la procesarea întrebării."}), 500

@app.route('/save_costs', methods=['POST'])
def save_costs():
    """Endpoint manual pentru salvarea costurilor"""
    try:
        save_to_excel()
        return jsonify({"status": "success", "message": "Costs saved to Excel"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # La oprirea serverului, salvează orice date rămase
        if usage_data:
            save_to_excel()