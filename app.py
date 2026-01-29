import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pickle
import os
import re
from pypdf import PdfReader
import hashlib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Multi-Resume Q&A System",
    page_icon="📄",
    layout="centered"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key"""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OpenAI API key not found!")
            st.stop()
        
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        st.stop()

client = get_openai_client()

# Initialize session state for storing resumes
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = {
        'chunks': [],
        'embeddings': [],
        'index': None,
        'resume_names': [],
        'chunk_to_resume': [],
        'rejected_files': []
    }

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Track current subject
if 'current_subject' not in st.session_state:
    st.session_state.current_subject = None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def validate_resume(text, filename):
    """
    Check if the uploaded PDF looks like a resume
    Returns: (is_valid, confidence_score, message, details)
    """
    if not text or len(text.strip()) < 100:
        return False, 0, "File is too short or empty", {
            'filename': filename,
            'reason': 'Empty or too short',
            'text_length': len(text) if text else 0
        }
    
    # Common resume indicators
    resume_keywords = [
        'experience', 'education', 'skills', 'work', 'project',
        'university', 'degree', 'bachelor', 'master', 'employment',
        'responsibilities', 'achievements', 'job', 'career'
    ]
    
    # Section headers (strong indicators)
    resume_sections = [
        'EDUCATION', 'WORK EXPERIENCE', 'EXPERIENCE', 'PROJECTS',
        'SKILLS', 'TECHNICAL SKILLS', 'SUMMARY', 'CERTIFICATIONS',
        'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT HISTORY'
    ]
    
    text_lower = text.lower()
    text_upper = text.upper()
    
    # Count keyword matches
    keyword_count = sum(1 for keyword in resume_keywords if keyword in text_lower)
    section_count = sum(1 for section in resume_sections if section in text_upper)
    
    # Calculate confidence
    confidence = 0
    indicators = []
    
    # Has multiple resume keywords
    if keyword_count >= 3:
        confidence += 30
        indicators.append(f"{keyword_count} resume keywords")
    
    # Has at least one section header
    if section_count >= 1:
        confidence += 40
        indicators.append(f"{section_count} section headers")
    
    # Has date patterns (years like 2019-2023)
    date_pattern = r'\b(19|20)\d{2}\b'
    dates = re.findall(date_pattern, text)
    if dates:
        confidence += 15
        indicators.append(f"{len(dates)} year mentions")
    
    # Has email pattern
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        confidence += 15
        indicators.append("Email found")
    
    # Build details
    details = {
        'filename': filename,
        'confidence': confidence,
        'text_length': len(text),
        'keyword_count': keyword_count,
        'section_count': section_count,
        'indicators': indicators
    }
    
    # Validation thresholds
    if confidence >= 60:
        return True, confidence, "Looks like a valid resume", details
    elif confidence >= 30:
        return True, confidence, "Possibly a resume, but structure may be unclear", details
    else:
        details['reason'] = f"Low confidence score ({confidence}%) - missing resume indicators"
        return False, confidence, f"Doesn't look like a resume (confidence: {confidence}%)", details

def fallback_chunk_by_size(text, resume_name, chunk_size=800, overlap=100):
    """
    Fallback chunking strategy when resume structure isn't detected
    Chunks by character count with overlap
    """
    chunks = []
    text_length = len(text)
    start = 0
    chunk_num = 0
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at paragraph
        if end < text_length:
            # Look for paragraph break within last 200 chars
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start + chunk_size - 200:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break > start + chunk_size - 200:
                    end = sentence_break + 1
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                'section': f'Chunk {chunk_num + 1}',
                'content': chunk_text,
                'resume_name': resume_name
            })
            chunk_num += 1
        
        start = end - overlap
    
    return chunks

def smart_chunk_resume(text, resume_name):
    """Chunk resume text into semantic sections"""
    chunks = []
    lines = text.split('\n')
    current_chunk = ""
    current_section = None
    found_sections = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        if line.upper() in ['EDUCATION', 'WORK EXPERIENCE', 'EXPERIENCE', 'PROJECTS', 
                           'TECHNICAL SKILLS', 'SKILLS', 'CERTIFICATIONS', 'SUMMARY',
                           'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT HISTORY']:
            found_sections = True
            if current_chunk and len(current_chunk) > 100:
                chunks.append({
                    'section': current_section,
                    'content': current_chunk.strip(),
                    'resume_name': resume_name
                })
            current_section = line
            current_chunk = ""
            continue
        
        # For EDUCATION section, keep everything together
        if current_section in ['EDUCATION', 'TECHNICAL SKILLS', 'SKILLS']:
            current_chunk += line + "\n"
            continue
        
        # Detect project headers
        is_project_header = (
            current_section in ['PROJECTS', 'PROJECT'] and 
            ':' in line and 
            any(tech in line for tech in ['Python', 'SQL', 'Java', 'React', 'Node', 'JavaScript', 'C++', 'Ruby']) and
            not line.startswith('•')
        )
        
        # Detect job headers
        is_job_header = (
            current_section in ['WORK EXPERIENCE', 'EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT HISTORY'] and
            ('|' in line or '–' in line or '-' in line) and
            re.search(r'\d{4}', line)
        )
        
        # If new entry and we have content, save current chunk
        if (is_project_header or is_job_header) and current_chunk:
            if len(current_chunk) > 100:
                chunks.append({
                    'section': current_section,
                    'content': current_chunk.strip(),
                    'resume_name': resume_name
                })
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    
    # Don't forget last chunk
    if current_chunk and len(current_chunk) > 100:
        chunks.append({
            'section': current_section,
            'content': current_chunk.strip(),
            'resume_name': resume_name
        })
    
    return chunks, found_sections

def process_resume(pdf_file, resume_name):
    """Process a single resume: extract text, chunk, and embed"""
    # Extract text
    text = extract_text_from_pdf(pdf_file)
    if not text:
        rejection_info = {
            'filename': resume_name,
            'reason': 'Could not extract text from PDF',
            'confidence': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.resume_data['rejected_files'].append(rejection_info)
        st.error(f"❌ {resume_name}: Could not extract text from PDF")
        return None
    
    # Validate if it's a resume
    is_valid, confidence, message, details = validate_resume(text, resume_name)
    
    if not is_valid:
        rejection_info = {
            'filename': resume_name,
            'reason': message,
            'confidence': confidence,
            'details': details,
            'text_preview': text[:500] + "..." if len(text) > 500 else text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.resume_data['rejected_files'].append(rejection_info)
        
        st.warning(f"⚠️ {resume_name}: {message}")
        st.info("📦 File stored but not processed. You can review rejected files in the sidebar.")
        return None
    
    if confidence < 60:
        st.warning(f"⚠️ {resume_name}: {message}")
        st.info("The file will be processed, but it may not have a standard resume format.")
    
    # Try semantic chunking first
    chunks, found_sections = smart_chunk_resume(text, resume_name)
    
    # Fallback to size-based chunking
    if not chunks or not found_sections:
        st.warning(f"⚠️ {resume_name}: No standard resume sections detected. Using fallback chunking.")
        chunks = fallback_chunk_by_size(text, resume_name)
    
    if not chunks:
        rejection_info = {
            'filename': resume_name,
            'reason': 'Could not create any chunks from this file',
            'confidence': confidence,
            'text_preview': text[:500] + "...",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.resume_data['rejected_files'].append(rejection_info)
        st.error(f"❌ {resume_name}: Could not create any chunks from this file.")
        return None
    
    st.info(f"📄 {resume_name}: Created {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = []
    progress = st.progress(0, text=f"Embedding {resume_name}...")
    
    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk['content']
            )
            embeddings.append(response.data[0].embedding)
            progress.progress((i + 1) / len(chunks))
        except Exception as e:
            rejection_info = {
                'filename': resume_name,
                'reason': f'Embedding failed at chunk {i}: {str(e)}',
                'confidence': confidence,
                'chunks_created': len(chunks),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.resume_data['rejected_files'].append(rejection_info)
            st.error(f"❌ Error embedding {resume_name}: {e}")
            progress.empty()
            return None
    
    progress.empty()
    
    return {
        'chunks': chunks,
        'embeddings': embeddings,
        'resume_name': resume_name
    }

def build_faiss_index(all_embeddings):
    """Build FAISS index from all embeddings"""
    if not all_embeddings:
        return None
    
    embeddings_array = np.array(all_embeddings).astype('float32')
    embedding_dim = embeddings_array.shape[1]
    
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    
    return index

def resolve_question_context(question, conversation_history, resume_names):
    """
    Resolve pronouns and implicit references in questions using conversation history
    Improved name matching for partial names and comparison detection
    
    Returns: (resolved_question, subject_name)
    """
    question_lower = question.lower()
    
    # Check if this is a comparison question
    comparison_keywords = ['and', 'vs', 'versus', 'compare', 'both', 'all', 'between', 'common', 'difference', 'differ']
    is_comparison = any(keyword in question_lower for keyword in comparison_keywords)
    
    # Count how many different people are mentioned
    mentioned_people = []
    for name in resume_names:
        name_base = name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
        name_parts = [part for part in name_base.split() if len(part) > 2]
        
        if any(part in question_lower for part in name_parts):
            mentioned_people.append(name_base)
    
    # If comparison question with multiple people, don't filter by person
    if is_comparison and len(mentioned_people) >= 2:
        return question, None  # Return None to prevent filtering
    
    # Pronouns and implicit references to check
    pronouns = ['he', 'she', 'they', 'his', 'her', 'their', 'him', 'them']
    implicit_refs = ['where did', 'when did', 'what did', 'how did', 'why did']
    
    has_pronoun = any(f' {pronoun} ' in f' {question_lower} ' for pronoun in pronouns)
    has_implicit = any(ref in question_lower for ref in implicit_refs)
    
    # Check if question explicitly mentions someone (for single-person questions)
    mentioned_person = None
    best_match_score = 0
    best_match_name = None
    
    for name in resume_names:
        name_base = name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
        name_parts = [part for part in name_base.split() if len(part) > 2]
        
        matches = 0
        for part in name_parts:
            if part in question_lower:
                matches += 1
        
        if matches > best_match_score:
            best_match_score = matches
            best_match_name = name_base
            mentioned_person = name
    
    # If we found a good match and it's NOT a comparison, update context
    if best_match_score > 0 and not is_comparison:
        return question, best_match_name
    
    # If has pronoun or implicit reference, look at conversation history
    if (has_pronoun or has_implicit) and conversation_history:
        for prev_q, prev_a in reversed(conversation_history):
            prev_q_lower = prev_q.lower()
            
            for name in resume_names:
                name_base = name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
                name_parts = [part for part in name_base.split() if len(part) > 2]
                
                if any(part in prev_q_lower for part in name_parts):
                    resolved_question = question
                    first_name = name_parts[0] if name_parts else name_base
                    
                    replacements = {
                        ' he ': f' {first_name} ',
                        ' she ': f' {first_name} ',
                        ' his ': f" {first_name}'s ",
                        ' her ': f" {first_name}'s ",
                        ' him ': f' {first_name} ',
                        ' them ': f' {first_name} ',
                        ' their ': f" {first_name}'s ",
                    }
                    
                    for pronoun, replacement in replacements.items():
                        if pronoun in f' {resolved_question.lower()} ':
                            resolved_question = resolved_question.lower().replace(pronoun, replacement)
                    
                    words_in_question = resolved_question.lower().split()
                    has_name_in_question = any(
                        any(part in word for part in name_parts)
                        for word in words_in_question
                    )
                    
                    if has_implicit and not has_name_in_question:
                        for ref in implicit_refs:
                            if resolved_question.lower().startswith(ref):
                                resolved_question = resolved_question.lower().replace(
                                    ref, 
                                    f"{ref} {first_name}"
                                )
                                break
                    
                    return resolved_question, name_base
    
    # No context found
    return question, None

def filter_chunks_by_person(chunks, indices, distances, person_name, resume_names):
    """
    Filter retrieved chunks to only include those from a specific person's resume
    FIXED: Proper numpy array handling
    """
    if not person_name:
        return indices, distances
    
    # Find the matching resume file
    matching_resume = None
    for resume_name in resume_names:
        resume_base = resume_name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
        
        # Check if person_name matches this resume
        if person_name in resume_base or resume_base in person_name:
            matching_resume = resume_name
            break
        
        # Also check if key parts of the name match
        person_parts = set(person_name.split())
        resume_parts = set(resume_base.split())
        
        if len(person_parts.intersection(resume_parts)) >= 2:
            matching_resume = resume_name
            break
    
    if not matching_resume:
        return indices, distances
    
    # Filter to only include chunks from this resume
    filtered_indices = []
    filtered_distances = []
    
    for idx, dist in zip(indices, distances):
        if chunks[idx]['resume_name'] == matching_resume:
            filtered_indices.append(idx)
            filtered_distances.append(dist)
    
    # If we filtered everything out, return original
    if not filtered_indices:
        return indices, distances
    
    # Convert to numpy arrays with proper dtype
    return (
        np.array(filtered_indices, dtype=np.int64),
        np.array(filtered_distances, dtype=np.float32)
    )

def ask_resume_rag(question, k=3):
    """
    RAG function to answer questions about uploaded resumes with conversation context
    """
    if st.session_state.resume_data['index'] is None:
        return "Please upload at least one resume first!", [], [], None
    
    # Resolve pronouns and references using conversation history
    resolved_question, subject = resolve_question_context(
        question,
        st.session_state.conversation_history,
        st.session_state.resume_data['resume_names']
    )
    
    # Update current subject if found
    if subject:
        st.session_state.current_subject = subject
    
    # Show if question was resolved
    question_modified = resolved_question != question
    
    # Embed the resolved question
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=resolved_question
    )
    
    # Search FAISS - adjust based on question type
    query_vector = np.array([query_response.data[0].embedding]).astype('float32')
    
    # Detect if it's a comparison question
    is_comparison = subject is None and any(
        keyword in question.lower() 
        for keyword in ['and', 'both', 'common', 'compare', 'versus', 'vs', 'difference']
    )
    
    # Increase k for comparison questions to get chunks from multiple people
    if is_comparison:
        search_k = min(k * 2, len(st.session_state.resume_data['chunks']))
    elif subject:
        search_k = min(k * 3, len(st.session_state.resume_data['chunks']))
    else:
        search_k = k
    
    distances, indices = st.session_state.resume_data['index'].search(query_vector, search_k)
    
    # Filter by person if we have a subject
    if subject:
        filtered_idx, filtered_dist = filter_chunks_by_person(
            st.session_state.resume_data['chunks'],
            indices[0],
            distances[0],
            subject,
            st.session_state.resume_data['resume_names']
        )
        indices_final = filtered_idx[:k]
        distances_final = filtered_dist[:k]
    else:
        indices_final = indices[0][:search_k]
        distances_final = distances[0][:search_k]
    
    # Gather context from retrieved chunks
    context_chunks = []
    resume_sources = []
    
    for idx in indices_final:
        chunk = st.session_state.resume_data['chunks'][idx]
        context_chunks.append(chunk['content'])
        resume_sources.append(chunk['resume_name'])
    
    # Build context with sources
    context = "\n\n---\n\n".join([
        f"[From {resume_sources[i]}]\n{context_chunks[i]}" 
        for i in range(len(context_chunks))
    ])
    
    # Add conversation history to context (last 3 exchanges)
    conversation_context = ""
    if st.session_state.conversation_history:
        recent_history = st.session_state.conversation_history[-3:]
        conversation_context = "\n\nPrevious conversation:\n"
        for prev_q, prev_a in recent_history:
            conversation_context += f"Q: {prev_q}\nA: {prev_a}\n\n"
    
    # Determine if question is about specific person or general
    is_specific_person = subject is not None
    
    # Create appropriate prompt
    if is_specific_person:
        prompt = f"""You are answering questions about a specific person's professional background in a natural, first-person conversational style.
{conversation_context}
Current Question: {resolved_question}

Relevant information:
{context}

Instructions:
- Answer in FIRST PERSON (use "I", "my", "I worked on") as if you are the person
- Keep your answer SHORT and CONCISE (2-3 sentences maximum)
- Only mention the most relevant skills/experience directly asked about
- Don't list every single detail - be selective
- If asking about skills, list 5-7 key ones, not everything
- Be conversational but brief

Answer naturally (keep it short):"""
        
        system_message = "You are answering as the person whose resume is being discussed. Speak in first person naturally and conversationally."
    else:
        # Check if this is a comparison question
        is_comparison_question = any(
            keyword in resolved_question.lower() 
            for keyword in ['common', 'both', 'compare', 'difference', 'versus', 'vs', 'and']
        )
        
        if is_comparison_question:
            prompt = f"""You are comparing professional backgrounds from multiple resumes.
{conversation_context}
Current Question: {resolved_question}

Available information from multiple resumes:
{context}

Instructions:
- For "what in common" questions: List ONLY the overlapping skills/experiences
- Look at ALL the resume sources provided and find commonalities
- Mention each person by name when noting commonalities
- Keep answer to 3-4 sentences
- Be specific about what they share

Answer with common items:"""
            system_message = "You are comparing multiple resumes to find commonalities or differences."
        else:
            prompt = f"""Answer questions about professional backgrounds based on the provided resume information.
{conversation_context}
Current Question: {resolved_question}

Available information from multiple resumes:
{context}

Instructions:
- Keep answer SHORT and CONCISE (2-4 sentences maximum)
- Mention each person by name clearly
- Be objective and factual
- Don't elaborate unless asked

Answer (keep it brief):"""
            system_message = "You are answering questions about professional backgrounds from multiple resumes."
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    answer = response.choices[0].message.content
    
    # Store in conversation history (use original question)
    st.session_state.conversation_history.append((question, answer))
    
    # Keep only last 10 exchanges
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]
    
    return answer, indices_final, distances_final, resolved_question if question_modified else None


# ==================== STREAMLIT UI ====================

# Header
st.title("📄 Multi-Resume Q&A System")
st.markdown("Upload up to 20 resumes and ask questions about them using AI-powered search.")

# Sidebar for uploads and management
with st.sidebar:
    st.header("📤 Upload Resumes")
    
    # Show current resume count
    current_count = len(st.session_state.resume_data['resume_names'])
    rejected_count = len(st.session_state.resume_data['rejected_files'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Processed", f"{current_count}/20")
    with col2:
        st.metric("⚠️ Rejected", rejected_count)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"You can upload up to {20 - current_count} more resumes"
    )
    
    # Process uploaded files
    if uploaded_files and st.button("Process Resumes", type="primary"):
        if current_count + len(uploaded_files) > 20:
            st.error(f"Maximum 20 resumes allowed! You're trying to upload {len(uploaded_files)} but only have {20 - current_count} slots left.")
        else:
            successful_uploads = 0
            failed_uploads = 0
            
            with st.spinner("Processing resumes..."):
                for uploaded_file in uploaded_files:
                    resume_name = uploaded_file.name
                    
                    # Check for duplicates
                    if resume_name in st.session_state.resume_data['resume_names']:
                        st.warning(f"'{resume_name}' already uploaded. Skipping.")
                        continue
                    
                    # Process resume
                    result = process_resume(uploaded_file, resume_name)
                    
                    if result:
                        st.session_state.resume_data['chunks'].extend(result['chunks'])
                        st.session_state.resume_data['embeddings'].extend(result['embeddings'])
                        st.session_state.resume_data['resume_names'].append(resume_name)
                        
                        st.success(f"✅ {resume_name}: Successfully processed!")
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                
                # Rebuild FAISS index
                if successful_uploads > 0:
                    st.session_state.resume_data['index'] = build_faiss_index(
                        st.session_state.resume_data['embeddings']
                    )
                    st.success(f"🎉 Successfully processed {successful_uploads} resume(s)!")
                
                if failed_uploads > 0:
                    st.info(f"📦 {failed_uploads} file(s) stored but not processed. See rejected files below.")
    
    # Show loaded resumes
    if st.session_state.resume_data['resume_names']:
        st.markdown("---")
        st.markdown("**✅ Processed Resumes:**")
        for i, name in enumerate(st.session_state.resume_data['resume_names'], 1):
            # Show simplified name
            display_name = name.replace('.pdf', '').replace('_', ' ')
            st.text(f"{i}. {display_name}")
    
    # Show rejected files
    if st.session_state.resume_data['rejected_files']:
        st.markdown("---")
        st.markdown("**⚠️ Rejected Files:**")
        
        with st.expander(f"View {rejected_count} rejected file(s)", expanded=False):
            for i, rejected in enumerate(st.session_state.resume_data['rejected_files'], 1):
                st.markdown(f"**{i}. {rejected['filename']}**")
                st.caption(f"🕐 {rejected.get('timestamp', 'Unknown time')}")
                st.caption(f"❌ Reason: {rejected['reason']}")
                
                if 'confidence' in rejected:
                    st.caption(f"📊 Confidence: {rejected['confidence']}%")
                
                if 'indicators' in rejected.get('details', {}):
                    indicators = rejected['details']['indicators']
                    if indicators:
                        st.caption(f"✓ Found: {', '.join(indicators)}")
                    else:
                        st.caption("✗ No resume indicators found")
                
                if 'text_preview' in rejected:
                    with st.expander("👁️ View text preview", expanded=False):
                        st.text(rejected['text_preview'])
                
                st.markdown("---")
            
            if st.button("🗑️ Clear Rejected Files", key="clear_rejected"):
                st.session_state.resume_data['rejected_files'] = []
                st.rerun()
    
    # Clear all button
    if st.session_state.resume_data['resume_names'] or st.session_state.resume_data['rejected_files']:
        st.markdown("---")
        if st.button("🗑️ Clear All (Processed + Rejected)", type="secondary"):
            st.session_state.resume_data = {
                'chunks': [],
                'embeddings': [],
                'index': None,
                'resume_names': [],
                'chunk_to_resume': [],
                'rejected_files': []
            }
            st.session_state.conversation_history = []
            st.session_state.current_subject = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    k_value = st.slider("Chunks to retrieve", 1, 10, 3)
    show_debug = st.checkbox("Show debug info", value=False)

# Main area
if not st.session_state.resume_data['resume_names']:
    st.info("👈 Upload resumes using the sidebar to get started!")
    
    # Show statistics if there are rejected files
    if st.session_state.resume_data['rejected_files']:
        st.warning(f"⚠️ You have {len(st.session_state.resume_data['rejected_files'])} rejected file(s). Check the sidebar for details.")
        
        # Show common rejection reasons
        reasons = {}
        for rejected in st.session_state.resume_data['rejected_files']:
            reason = rejected.get('reason', 'Unknown')
            if 'confidence' in reason.lower():
                reason = "Low confidence (not resume-like)"
            reasons[reason] = reasons.get(reason, 0) + 1
        
        st.markdown("**📊 Rejection Summary:**")
        for reason, count in reasons.items():
            st.markdown(f"- {reason}: **{count}** file(s)")
    
    st.markdown("""
    ### How to use:
    1. Upload 1-20 PDF resumes using the sidebar
    2. Click "Process Resumes" to build the search system
    3. Ask questions about the resumes
    
    ### Example questions:
    - "What experience does [Name] have with Python?"
    - "Who has machine learning experience?"
    - "Compare the educational backgrounds"
    - "Which candidates have worked at startups?"
    
    ### ℹ️ File Requirements:
    Files should be resume/CV PDFs containing:
    - Standard sections (Education, Experience, Skills, etc.)
    - Resume keywords (employment, responsibilities, projects)
    - Contact information (email)
    - Dates (work history, education)
    
    **Note:** Files that don't meet these criteria will be stored but not processed for searching. You can review them in the sidebar.
    """)
else:
    # Show conversation history ALWAYS VISIBLE
    if st.session_state.conversation_history:
        st.markdown("### 💬 Full Conversation")
        
        # Show all conversations inline
        for i, (q, a) in enumerate(st.session_state.conversation_history, 1):
            # Question
            st.markdown(f"**Q{i}:** {q}")
            # Answer in info box
            st.info(f"**A{i}:** {a}")
            st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation History", key="clear_conv_main"):
            st.session_state.conversation_history = []
            st.session_state.current_subject = None
            st.rerun()
        
        st.markdown("---")
    
    # Show current context
    if st.session_state.current_subject:
        st.info(f"📍 Current context: Talking about **{st.session_state.current_subject.title()}**")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        # Get first resume name for examples
        first_resume = st.session_state.resume_data['resume_names'][0].replace('.pdf', '').replace('_', ' ').split()[0]
        
        st.markdown(f"""
        **About specific people:**
        - "What did {first_resume} do at their last job?"
        - "What machine learning projects has [name] worked on?"
        - "Tell me about [name]'s educational background"
        
        **Follow-up questions (uses context):**
        - After asking about someone: "Where did he/she get their degree?"
        - After asking about someone: "What projects did he/she work on?"
        - "When did he/she graduate?"
        - "What technologies does he/she know?"
        
        **Comparing candidates:**
        - "Who has the most Python experience?"
        - "Compare the educational backgrounds"
        - "Which candidates have startup experience?"
        - "Who knows AWS?"
        - "What skills do [Name1] and [Name2] have in common?"
        """)
    
    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What skills do John and Sarah have in common?",
        help="Ask about specific people or compare across all resumes. I'll remember the context!"
    )
    
    # Answer
    if st.button("🔍 Get Answer", type="primary") or question:
        if question:
            with st.spinner("🤔 Searching resumes and generating answer..."):
                try:
                    answer, retrieved_indices, distances, resolved_q = ask_resume_rag(question, k=k_value)
                    
                    # Show if question was resolved using context
                    if resolved_q:
                        st.info(f"💡 Interpreted as: *{resolved_q}*")
                    
                    st.success("✅ Answer:")
                    st.markdown(answer)
                    
                    # Show sources
                    if len(retrieved_indices) > 0:
                        with st.expander("📚 Sources Used"):
                            for i, idx in enumerate(retrieved_indices):
                                chunk = st.session_state.resume_data['chunks'][idx]
                                st.markdown(f"**Source {i+1}:** {chunk['resume_name']} - {chunk['section']}")
                                st.caption(f"Relevance score: {1 / (1 + distances[i]):.3f}")
                    
                    # Debug info
                    if show_debug:
                        with st.expander("🔍 Debug Info"):
                            st.write(f"**Current subject:** {st.session_state.current_subject}")
                            st.write(f"**Conversation length:** {len(st.session_state.conversation_history)} exchanges")
                            st.write(f"**Question resolved:** {resolved_q is not None}")
                            
                            for i, idx in enumerate(retrieved_indices):
                                chunk = st.session_state.resume_data['chunks'][idx]
                                st.markdown(f"**Chunk {i+1}**")
                                st.write(f"Resume: {chunk['resume_name']}")
                                st.write(f"Section: {chunk['section']}")
                                st.write(f"Distance: {distances[i]:.4f}")
                                st.code(chunk['content'][:200] + "...", language="text")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if show_debug:
                        st.exception(e)
        else:
            st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Multi-Resume RAG System with Smart Comparison & Full Chat History | Powered by OpenAI, FAISS, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)