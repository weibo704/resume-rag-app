import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Weibo's Resume Q&A",
    page_icon="💼",
    layout="centered"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key"""
    try:
        # Try to get from Streamlit secrets first, then environment variable
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OpenAI API key not found! Please set it in .streamlit/secrets.toml or as environment variable.")
            st.stop()
        
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        st.stop()

client = get_openai_client()

# Load or build RAG system
@st.cache_resource
def load_or_build_rag_system():
    """Load existing RAG system or build from resume.pdf"""
    import re
    from pypdf import PdfReader
    
    # Check if pre-built files exist
    if os.path.exists('resume_chunks.pkl') and os.path.exists('resume_faiss.index'):
        try:
            with open('resume_chunks.pkl', 'rb') as f:
                chunks = pickle.load(f)
            index = faiss.read_index('resume_faiss.index')
            st.success("✅ Loaded pre-built RAG system")
            return chunks, index
        except:
            pass  # Fall through to rebuild
    
    # Build from resume.pdf
    st.info("📦 Building RAG system from resume.pdf... This will take about 30 seconds.")
    
    def smart_chunk_resume(pdf_path):
        """Chunk resume by sections"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line in ['EDUCATION', 'WORK EXPERIENCE', 'PROJECTS', 'TECHNICAL SKILLS']:
                if current_chunk and len(current_chunk) > 100:
                    chunks.append({
                        'section': current_section,
                        'content': current_chunk.strip()
                    })
                current_section = line
                current_chunk = ""
                continue
            
            if current_section == "EDUCATION":
                current_chunk += line + "\n"
                continue
            
            is_project_header = (
                current_section == "PROJECTS" and 
                ':' in line and 
                ('Python' in line or 'SQL' in line) and
                not line.startswith('•')
            )
            
            is_job_header = (
                current_section == "WORK EXPERIENCE" and
                '|' in line and
                re.search(r'\d{4}', line)
            )
            
            if (is_project_header or is_job_header) and current_chunk:
                if len(current_chunk) > 100:
                    chunks.append({
                        'section': current_section,
                        'content': current_chunk.strip()
                    })
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        if current_chunk and len(current_chunk) > 100:
            chunks.append({
                'section': current_section,
                'content': current_chunk.strip()
            })
        
        return chunks
    
    # Chunk resume
    chunks = smart_chunk_resume("resume.pdf")
    
    # Create embeddings
    embeddings = []
    progress_bar = st.progress(0, text="Creating embeddings...")
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['content']
        )
        embeddings.append(response.data[0].embedding)
        progress_bar.progress((i + 1) / len(chunks), text=f"Embedding chunk {i+1}/{len(chunks)}")
    
    progress_bar.empty()
    
    # Build FAISS index
    embeddings_array = np.array(embeddings).astype('float32')
    embedding_dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    
    st.success("✅ RAG system built successfully!")
    
    return chunks, index

# Load RAG system
chunks, index = load_or_build_rag_system()

def ask_resume_rag(question, k=3):
    """
    RAG function to answer questions about Weibo's resume
    
    Args:
        question: User's question
        k: Number of chunks to retrieve
    
    Returns:
        Generated answer, retrieved indices, distances
    """
    # Step 1: Embed the question
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    
    # Step 2: Search FAISS for similar chunks
    query_vector = np.array([query_response.data[0].embedding]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    # Step 3: Gather context from retrieved chunks
    context_chunks = [chunks[idx]['content'] for idx in indices[0]]
    context = "\n\n".join(context_chunks)
    
    # Step 4: Create prompt for LLM
    prompt = f"""Based on the following information about Weibo, answer this question: {question}

Context:
{context}

IMPORTANT: 
- Only use information that DIRECTLY answers the question
- Do NOT combine information from different jobs or time periods
- If a project or experience is not explicitly connected to what's being asked, do not mention it
- Be specific about which role or project you're referring to

Answer:"""
    
    # Step 5: Generate answer using GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about Weibo's professional background based on his resume."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content, indices[0], distances[0]


# ==================== STREAMLIT UI ====================

# Header
st.title("💼 Ask About Weibo's Resume")
st.markdown("Get AI-powered answers about Weibo's professional background, education, and projects.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **Retrieval-Augmented Generation (RAG)** to intelligently answer questions about Weibo's resume.
    
    **How it works:**
    1. 📄 Resume is split into semantic chunks
    2. 🔍 Your question is matched using AI embeddings
    3. 🤖 GPT generates a focused answer
    
    **Technology:**
    - OpenAI Embeddings (text-embedding-3-small)
    - FAISS Vector Search
    - GPT-4o-mini
    - Streamlit
    """)
    
    st.markdown("---")
    st.markdown("**📊 System Stats**")
    st.metric("Resume Chunks", len(chunks))
    st.metric("Vectors Indexed", index.ntotal)
    
    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    k_value = st.slider("Chunks to retrieve (k)", 1, 5, 3, 
                       help="Higher values retrieve more context but may include less relevant info")
    show_debug = st.checkbox("Show debug info", value=False)

# Example questions
with st.expander("💡 Example Questions"):
    examples = [
        "What did Weibo do at Hoolii?",
        "What machine learning projects has Weibo worked on?",
        "Where did Weibo go to school?",
        "What is Weibo's educational background?",
        "Tell me about Weibo's LLM orchestration project",
        "What technologies does Weibo know?",
        "What AWS experience does Weibo have?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.question = example

# Main question input
question = st.text_input(
    "Your question:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., What did Weibo do at Hoolii?",
    help="Ask anything about Weibo's professional background"
)

# Answer button
if st.button("🔍 Get Answer", type="primary") or question:
    if question:
        with st.spinner("🤔 Searching resume and generating answer..."):
            try:
                # Get answer
                answer, retrieved_indices, distances = ask_resume_rag(question, k=k_value)
                
                # Display answer
                st.success("✅ Answer:")
                st.markdown(answer)
                
                # Show debug information if enabled
                if show_debug:
                    with st.expander("🔍 Debug: Retrieved Chunks"):
                        for i, (idx, dist) in enumerate(zip(retrieved_indices, distances)):
                            st.markdown(f"**Chunk {i+1}** - Distance: `{dist:.4f}` - Section: `{chunks[idx]['section']}`")
                            st.code(chunks[idx]['content'][:300] + "...", language="text")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your OpenAI API key is configured correctly.")
                
                if show_debug:
                    st.exception(e)
    else:
        st.warning("⚠️ Please enter a question!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Powered by OpenAI, FAISS, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)