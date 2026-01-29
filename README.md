# Multi-Resume Q&A System

An AI-powered Resume Q&A system that allows you to upload up to 20 resumes and ask questions about them using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- **Multi-Resume Upload**: Upload and process up to 20 PDF resumes simultaneously
- **Intelligent Q&A**: Ask questions about specific candidates or compare multiple candidates
- **Conversation Memory**: System remembers context and handles follow-up questions
- **Smart Name Matching**: Handles partial names and variations (e.g., "John" matches "John Smith")
- **Resume Validation**: Automatically validates uploaded files and stores rejected ones for review
- **Source Citations**: See which resume each answer comes from
- **First-Person Responses**: When asking about a specific person, get answers in their voice

## 🚀 Live Demo

[Try it here](https://your-app-url.streamlit.app) *(Add your Streamlit Cloud URL after deployment)*

## 📋 How to Use

1. **Upload Resumes**: Click "Upload PDF resumes" in the sidebar
2. **Process**: Click "Process Resumes" to build the search system
3. **Ask Questions**: Type your question and click "Get Answer"

### Example Questions

**About specific people:**
- "What did John do at Google?"
- "What machine learning projects has Sarah worked on?"
- "Tell me about Mike's educational background"

**Comparison questions:**
- "What skills do John and Sarah have in common?"
- "Who has the most Python experience?"
- "Compare their educational backgrounds"

**Follow-up questions:**
- After asking about John: "Where did he get his degree?"
- After asking about Sarah: "What projects did she work on?"

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **LLM**: GPT-4o-mini
- **PDF Processing**: pypdf
- **Language**: Python 3.8+

## 📦 Installation & Local Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/resume-rag-app.git
cd resume-rag-app
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key

Create `.streamlit/secrets.toml`:
```bash
mkdir .streamlit
nano .streamlit/secrets.toml
```

Add your OpenAI API key:
```toml
OPENAI_API_KEY = "sk-proj-your-api-key-here"
```

### Step 5: Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🏗️ Architecture

### RAG Pipeline
```
1. PDF Upload → Extract Text
2. Text → Semantic Chunking (by sections)
3. Chunks → OpenAI Embeddings (1536-dimensional vectors)
4. Embeddings → FAISS Index (vector database)
5. User Question → Embed → Search FAISS → Retrieve Relevant Chunks
6. Chunks + Question → GPT-4o-mini → Generate Answer
```

### Key Components

- **Chunking**: Smart semantic chunking that preserves resume structure (education, work experience, projects as separate chunks)
- **Embedding**: OpenAI's text-embedding-3-small model converts text to 1536-dimensional vectors
- **Vector Search**: FAISS performs fast similarity search to find relevant resume sections
- **Context Resolution**: Tracks conversation history to resolve pronouns ("he", "she") to actual names
- **Filtering**: When asking about a specific person, filters results to only that resume

## 📊 System Capabilities

- **Resume Validation**: Checks if uploaded files are actually resumes (60%+ confidence required)
- **Fallback Chunking**: If standard resume structure isn't detected, uses size-based chunking
- **Rejected File Storage**: Invalid files are stored but not processed, with detailed rejection reasons
- **Conversation Context**: Remembers last 10 Q&A exchanges for follow-up questions
- **Name Resolution**: "Dhruv Kumar" matches "Dhruv Kamalesh Kumar GenAI Resume.pdf"

## 🔒 Privacy & Security

- ⚠️ **Never commit `.streamlit/secrets.toml` to GitHub** (contains API key)
- ⚠️ Resumes are processed in-session only (not permanently stored)
- ⚠️ When session ends, all resume data is cleared

## 💰 Cost Estimation

Using OpenAI's pricing (as of Jan 2026):

- **text-embedding-3-small**: $0.02 per 1M tokens
- **gpt-4o-mini**: $0.150 per 1M input tokens, $0.600 per 1M output tokens

**Example costs:**
- Processing 1 resume (~2000 tokens): ~$0.00004
- 1 question + answer: ~$0.0001
- Processing 20 resumes + 50 questions: ~$0.01

## 📝 Project Structure
```
resume-rag-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
└── .streamlit/
    └── secrets.toml         # OpenAI API key (not committed)
```

## 🚀 Deployment on Streamlit Cloud

1. Push your code to GitHub (this repository)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select this repository
5. Set main file: `app.py`
6. Add your OpenAI API key in "Advanced settings" → "Secrets"
7. Deploy!

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install -r requirements.txt
```

### "OpenAI API key not found"
Make sure `.streamlit/secrets.toml` exists and contains your API key

### "File doesn't look like a resume"
The file needs standard resume sections (EDUCATION, WORK EXPERIENCE, etc.) and resume keywords. Check the rejected files section for details.

### Comparison questions not working
Make sure you're using keywords like "and", "both", "common" in your question. Example: "What skills do John and Sarah have in common?"

## 🎓 Learning Outcomes

This project demonstrates:
- Retrieval-Augmented Generation (RAG) architecture
- Embedding-based semantic search
- Vector database implementation (FAISS)
- Multi-document question answering
- Conversation context management
- Production-ready error handling
- Web application development with Streamlit

## 📄 License

MIT License - feel free to use and modify!

## 👤 Author

**Weibo Zheng**
- MS Data Analytics Engineering @ Northeastern University
- [LinkedIn](https://linkedin.com/in/your-profile)
- [GitHub](https://github.com/weibo704)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI](https://openai.com/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)

## 📧 Contact

For questions or feedback, reach out at zheng.weib@northeastern.edu
