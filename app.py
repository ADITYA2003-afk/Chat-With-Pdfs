import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template
from PIL import Image
import io
import re
import tempfile
import os
import speech_recognition as sr
import fitz  # pymupdf
import pyttsx3
import requests
import base64
import time
import logging
from openai import OpenAI
from auth import Auth


# Initialize authentication
auth = Auth()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")

# Initialize recognizer, TTS engine, and OpenAI client
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def init_session_state():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "full_text" not in st.session_state:
        st.session_state.full_text = ""
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
    if "extracted_images" not in st.session_state:
        st.session_state.extracted_images = []
    if "selected_image_index" not in st.session_state:
        st.session_state.selected_image_index = None
    # New session state variables for chat history
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = 0
    if "conversation_names" not in st.session_state:
        st.session_state.conversation_names = ["New Chat"]

def login_page():
    st.title("Login to Chat with PDF ")
    
    # Initialize session state if not set
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    # Create a clean, modern login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("Login", use_container_width=True)
    
    if submit:
        if username and password:
            user = auth.login_user(username, password)  # Assuming auth is defined
            if user:
                st.session_state.user = user
                st.session_state.page = "main"
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")
    
    # Add register link/button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Register", key="go_to_register", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

def register_page():
    st.title("Register for Chat with pdf")
    
    # Initialize session state if not set
    if "page" not in st.session_state:
        st.session_state.page = "register"
    
    # Create a clean, modern registration form
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("Register", use_container_width=True)
    
    if submit:
        if username and email and password and confirm_password:
            # Basic email validation
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Invalid email format")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                if auth.register_user(username, email, password):  # Assuming auth is defined
                    st.success("Registration successful! Please login.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Username or email already exists")
        else:
            st.warning("Please fill in all fields")
    
    # Add login link/button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Login", key="go_to_login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

def extract_images_from_pdfs(pdf_docs):
    """Extract images from uploaded PDF documents using pymupdf."""
    images = []
    try:
        for pdf in pdf_docs:
            pdf_reader = fitz.open(stream=pdf.read(), filetype="pdf")
            for page_index in range(len(pdf_reader)):
                for img in pdf_reader[page_index].get_images(full=True):
                    xref = img[0]
                    base_image = pdf_reader.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
            pdf_reader.close()
    except Exception as e:
        st.error(f"Error extracting images: {e}")
    return images

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs using PyPDF2."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDFs: {e}")
    return text.strip()

def get_text_chunks(text):
    """Split text into chunks for vectorization."""
    if not text:
        return []
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def get_vectorstore(text_chunks):
    """Create a FAISS vectorstore from text chunks."""
    if not text_chunks:
        st.error("No text chunks available for vectorization.")
        return None
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Initialize a conversational chain with memory."""
    if not vectorstore:
        return None
    try:
        llm = ChatOpenAI(temperature=0.7)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def handle_userinput(user_question):
    """Handle user questions and display chat history."""
    if not st.session_state.get('conversation'):
        st.error("Please process documents first!")
        return
    try:
        response = st.session_state.conversation({'question': user_question})
        chat_history = response['chat_history']
        
        # Store the conversation
        if len(st.session_state.conversations) <= st.session_state.current_conversation:
            st.session_state.conversations.append([])
            st.session_state.conversation_names.append(f"Chat {len(st.session_state.conversations)}")
        
        # Add the new messages to the current conversation
        st.session_state.conversations[st.session_state.current_conversation].extend([
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": chat_history[-1].content}
        ])
        
        # Display the current conversation
        for message in st.session_state.conversations[st.session_state.current_conversation]:
            template = user_template if message["role"] == "user" else bot_template
            st.write(template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing question: {e}")

def speech_to_text():
    """Convert speech to text using microphone input."""
    try:
        with sr.Microphone() as source:
            st.info("Listening... Please speak.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
    return ""

def text_to_speech(text, save_file=False, filename="output.mp3"):
    """Convert text to speech and optionally save as an audio file."""
    if not text:
        st.warning("No text provided for text-to-speech.")
        return None
    try:
        tts_engine.setProperty('rate', 150)
        if save_file:
            temp_file = os.path.join(tempfile.gettempdir(), filename)
            tts_engine.save_to_file(text, temp_file)
            tts_engine.runAndWait()
            return temp_file
        else:
            tts_engine.say(text)
            tts_engine.runAndWait()
        return None
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def generate_image(prompt, size="1024x1024", max_retries=3):
    """Generate an image using OpenAI's DALL-E API via the OpenAI client."""
    if not client.api_key:
        st.error("OpenAI API key not found. Please set it in your .env file.")
        return None
    
    valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    if size not in valid_sizes:
        st.warning(f"Invalid size {size} for DALL-E 3. Defaulting to 1024x1024.")
        size = "1024x1024"
    
    logger.info(f"Generating image with prompt: {prompt}, size: {size}")

    for attempt in range(max_retries):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            logger.info(f"Received image URL: {image_url}")
            
            # Fetch the image from the URL
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            image = Image.open(io.BytesIO(image_response.content))
            return image
        except Exception as e:
            error_str = str(e)
            logger.error(f"Attempt {attempt + 1}/{max_retries} - Error: {error_str}")
            if "400" in error_str and "invalid_size" in error_str:
                st.error("Invalid size specified for DALL-E 3. Supported sizes are 1024x1024, 1792x1024, or 1024x1792.")
                return None
            elif "500" in error_str or "server error" in error_str.lower():
                if attempt < max_retries - 1:
                    st.warning(f"Server error. Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    st.error(f"Failed after {max_retries} attempts: Server Error. Please try again later or check OpenAI status.")
            elif "401" in error_str or "authentication" in error_str.lower():
                st.error("Authentication error (401): Invalid API key. Check your OPENAI_API_KEY in .env.")
                return None
            elif "429" in error_str or "rate limit" in error_str.lower():
                wait_time = 60  # Wait 1 minute for rate limit reset (adjust based on your plan)
                if attempt < max_retries - 1:
                    st.warning(f"Rate limit exceeded (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"Rate limit exceeded (429) after {max_retries} attempts. Please wait a few minutes and try again, or check your OpenAI plan for higher limits.")
                    st.info("You can check your rate limits and usage at: https://platform.openai.com/account/rate-limits")
                return None
            else:
                st.error(f"Error generating image: {error_str}")
                return None
    return None

def main():
    """Main application function."""
    st.write(css, unsafe_allow_html=True)
    init_session_state()
    
    # Handle authentication pages
    if st.session_state.page == "login":
        login_page()
        return
    elif st.session_state.page == "register":
        register_page()
        return
    
    # Main application (only accessible when logged in)
    if not st.session_state.user:
        st.session_state.page = "login"
        st.rerun()
        return
    
    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.user['username']}!")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()
        
        st.subheader("📤 Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True, type="pdf")
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF!")
            else:
                with st.spinner("Processing"):
                    # Get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.text_chunks = text_chunks
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success("Documents processed successfully!")

        # Image extraction section
        st.subheader("🖼️ Extract Images")
        image_pdf_docs = st.file_uploader("Upload PDFs for images", accept_multiple_files=True, type="pdf", key="image_uploader")
        if st.button("Extract Images"):
            if not image_pdf_docs:
                st.error("Please upload PDFs for image extraction!")
            else:
                with st.spinner("Processing PDFs..."):
                    st.session_state.extracted_images = extract_images_from_pdfs(image_pdf_docs)
                    if st.session_state.extracted_images:
                        st.session_state.selected_image_index = None
                        st.success(f"Extracted {len(st.session_state.extracted_images)} images.")
                    else:
                        st.info("No images found in uploaded PDFs.")

    # Main chat interface
    st.header("Chat with Multiple PDFs")
    
    # Chat history tabs
    col1, col2 = st.columns([4, 1])
    with col1:
        # Create tabs for conversations
        tabs = st.tabs(st.session_state.conversation_names)
        with tabs[st.session_state.current_conversation]:
            # Display current conversation
            if st.session_state.conversations and len(st.session_state.conversations) > st.session_state.current_conversation:
                for message in st.session_state.conversations[st.session_state.current_conversation]:
                    template = user_template if message["role"] == "user" else bot_template
                    st.write(template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    
    with col2:
        if st.button("➕ New Chat"):
            st.session_state.conversations.append([])
            st.session_state.conversation_names.append(f"Chat {len(st.session_state.conversations)}")
            st.session_state.current_conversation = len(st.session_state.conversations) - 1
            st.rerun()
    
    # Chat input with speech-to-text integration
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Ask a question about your documents:", key="question_input")
    with col2:
        if st.button("🎤 Speak to Ask"):
            recognized_text = speech_to_text()
            if recognized_text:
                st.session_state.question_input = recognized_text
                st.rerun()
    
    if user_question:
        handle_userinput(user_question)

    # Show processed text if available
    if st.session_state.text_chunks:
        with st.expander("View Processed Text"):
            text = "\n".join(st.session_state.text_chunks)
            st.text_area("", text, height=300)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔊 Read Text"):
                    text_to_speech(text)
            with col2:
                if st.button("💾 Save Audio"):
                    audio_file = text_to_speech(text, save_file=True)
                    if audio_file:
                        with open(audio_file, "rb") as f:
                            st.download_button(
                                "Download Audio",
                                f,
                                file_name="text_to_speech.mp3",
                                mime="audio/mp3"
                            )

    # Display extracted images
    if st.session_state.extracted_images:
        st.subheader("Extracted Images")
        for i, img in enumerate(st.session_state.extracted_images):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
            with col2:
                if st.button(f"Select Image {i+1}", key=f"select_{i}"):
                    st.session_state.selected_image_index = i
                    st.success(f"Selected Image {i+1}")

        # Image generation from selected image
        if st.session_state.selected_image_index is not None:
            st.subheader("Generate New Image")
            prompt = st.text_input(
                "Enter prompt for image generation:",
                placeholder="e.g., 'A modern version of this scene'"
            )
            if st.button("Generate Image"):
                if prompt:
                    with st.spinner(f"Generating image from prompt: {prompt}"):
                        image = generate_image(prompt)
                        if image:
                            st.session_state.generated_image = image
                            st.image(image, caption=f"Generated: {prompt}")
                            
                            # Save image option
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            st.download_button(
                                "Download Generated Image",
                                img_byte_arr.getvalue(),
                                file_name=f"generated_image.png",
                                mime="image/png"
                            )
                else:
                    st.warning("Please enter a prompt for image generation.")

if __name__ == "__main__":
    main()