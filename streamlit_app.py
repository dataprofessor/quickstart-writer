import streamlit as st
import requests
import os
from urllib.parse import urlparse, parse_qs
import json
import nbformat
from nbconvert import MarkdownExporter
import assemblyai as aai
import time
import re
import markdown
import yt_dlp
from openai import OpenAI
import anthropic
import zipfile
import io
import base64

# Initialize session state variables for managing application state
if 'notebook_state' not in st.session_state:
    st.session_state.notebook_state = {
        'content': None,
        'filename': None
    }
if 'show_error' not in st.session_state:
    st.session_state.show_error = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'markdown_content' not in st.session_state:
    st.session_state.markdown_content = None
if 'transcript_content' not in st.session_state:
    st.session_state.transcript_content = None
if 'generated_blog' not in st.session_state:
    st.session_state.generated_blog = None
if 'html_content' not in st.session_state:
    st.session_state.html_content = None
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'github_url' not in st.session_state:
    st.session_state.github_url = ""
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""
if 'notebook_input_method' not in st.session_state:
    st.session_state.notebook_input_method = "Upload File"
if 'content_input_method' not in st.session_state:
    st.session_state.content_input_method = "Upload File"

# Verify API keys are configured
if 'AAI_KEY' not in st.secrets:
    st.error("Please set your AssemblyAI API key in Streamlit secrets as AAI_KEY")
    st.stop()
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("Please set your OpenAI API key in Streamlit secrets as OPENAI_API_KEY")
    st.stop()

# Initialize API clients
aai.settings.api_key = st.secrets['AAI_KEY']

def remove_metadata_header(content):
    """Remove metadata header from markdown content."""
    # Find the first occurrence of a markdown heading (# )
    first_heading_match = re.search(r'^#\s+', content, re.MULTILINE)
    
    if first_heading_match:
        # Get the position where the first heading starts
        start_pos = first_heading_match.start()
        
        # Check if there's content before the first heading
        prefix = content[:start_pos].strip()
        
        # If there's prefix content and it looks like metadata, remove it
        if prefix and any(field in prefix.lower() for field in ['author:', 'id:', 'summary:', 'categories:', 'environments:', 'status:', 'tags:']):
            return content[start_pos:]
    
    return content

def remove_duration_markers(content):
    """Remove duration markers from markdown content."""
    # Remove lines that only contain duration information
    content = re.sub(r'^Duration:\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    # Remove duration from section headers
    content = re.sub(r'^(#+\s+.*?)\s*Duration:\s*\d+\s*$', r'\1', content, flags=re.MULTILINE)
    
    # Clean up any empty lines that might have been created
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def preprocess_markdown_content(content):
    """Apply all preprocessing steps to the markdown content."""
    if content is None:
        return None
        
    # Remove metadata header
    content = remove_metadata_header(content)
    
    # Remove duration markers
    content = remove_duration_markers(content)
    
    return content.strip()

def submit_callback():
    """Handle the submit button click"""
    st.session_state.submitted = True

def download_callback():
    """Handle the download action without triggering app reset"""
    return st.session_state.generated_blog

def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube or YouTube Shorts URL"""
    patterns = [
        r'https?:\/\/(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)(?:\?.*)?$',  # Short URLs
        r'https?:\/\/(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(?:&.*)?$',  # Regular watch URLs
        r'https?:\/\/(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)(?:\?.*)?$'  # Shorts URLs
    ]
    
    for pattern in patterns:
        if re.match(pattern, url.strip()):
            return True
    return False

def is_valid_github_url(url):
    """Validate GitHub URL for Jupyter notebook"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        return (
            parsed.netloc == "github.com" and
            len(parsed.path.split('/')) >= 3 and
            ("blob" in parsed.path or "tree" in parsed.path or ".ipynb" in parsed.path)
        )
    except:
        return False

def get_notebook_content(github_url):
    """Fetch Jupyter notebook content from GitHub"""
    try:
        parsed = urlparse(github_url)
        path_parts = parsed.path.split('/')
        
        if 'blob' in path_parts:
            blob_index = path_parts.index('blob')
            path_parts.pop(blob_index)
            path_parts.insert(blob_index, 'refs/heads')
        elif 'tree' in path_parts:
            tree_index = path_parts.index('tree')
            path_parts.pop(tree_index)
            path_parts.insert(tree_index, 'refs/heads')
        
        raw_url = f"https://raw.githubusercontent.com{'/'.join(path_parts)}"
        
        if not raw_url.endswith('.ipynb'):
            raw_url += '.ipynb'
        
        response = requests.get(raw_url)
        response.raise_for_status()
        
        return response.content, os.path.basename(raw_url)
        
    except Exception as e:
        st.session_state.error_message = f"Error: {str(e)}"
        st.session_state.show_error = True
        return None, None

def get_file_content(github_url):
    """Fetch content from GitHub"""
    try:
        parsed = urlparse(github_url)
        path_parts = parsed.path.split('/')
        
        if 'blob' in path_parts:
            blob_index = path_parts.index('blob')
            path_parts.pop(blob_index)
            path_parts.insert(blob_index, 'refs/heads')
        elif 'tree' in path_parts:
            tree_index = path_parts.index('tree')
            path_parts.pop(tree_index)
            path_parts.insert(tree_index, 'refs/heads')
        
        raw_url = f"https://raw.githubusercontent.com{'/'.join(path_parts)}"
        response = requests.get(raw_url)
        response.raise_for_status()
        
        filename = os.path.basename(raw_url)
        if filename.endswith('.ipynb'):
            notebook_json = json.loads(response.content)
            markdown_exporter = MarkdownExporter()
            content, _ = markdown_exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_json), as_version=4))
            return preprocess_markdown_content(content)  # Apply preprocessing
        else:
            content = response.content.decode('utf-8')
            return preprocess_markdown_content(content)  # Apply preprocessing
        
    except Exception as e:
        st.session_state.error_message = f"Error: {str(e)}"
        st.session_state.show_error = True
        return None

def on_url_change():
    """Handle GitHub URL changes and trigger notebook processing"""
    st.session_state.show_error = False
    st.session_state.error_message = ""
    
    if not st.session_state.github_url:
        st.session_state.notebook_state['content'] = None
        st.session_state.notebook_state['filename'] = None
        st.session_state.markdown_content = None
        return
    
    if is_valid_github_url(st.session_state.github_url):
        content, filename = get_notebook_content(st.session_state.github_url)
        if content:
            st.session_state.notebook_state['content'] = content
            st.session_state.notebook_state['filename'] = filename
            
            try:
                notebook_json = json.loads(content)
                markdown_content = convert_notebook_to_markdown(notebook_json)
                st.session_state.markdown_content = preprocess_markdown_content(markdown_content)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing notebook: {str(e)}")
            except Exception as e:
                st.error(f"Error converting to markdown: {str(e)}")
    else:
        st.session_state.error_message = "Invalid GitHub URL. Please provide a valid GitHub URL pointing to a Jupyter notebook (.ipynb file)."
        st.session_state.show_error = True
        st.session_state.notebook_state['content'] = None
        st.session_state.notebook_state['filename'] = None
        st.session_state.markdown_content = None

def download_audio(url, progress_bar):
    """Download YouTube video audio with progress tracking"""
    def progress_hook(d):
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes')
            downloaded_bytes = d.get('downloaded_bytes', 0)
            
            if total_bytes:
                progress = (downloaded_bytes / total_bytes) * 100
                progress_bar.progress(int(progress), text=f"Downloading... {int(progress)}%")
        elif d['status'] == 'finished':
            progress_bar.progress(100, text="Converting to WAV...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',
        'verbose': True,
        'progress_hooks': [progress_hook],
    }

    try:
        progress_bar.progress(0, text="Starting download...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        progress_bar.progress(100, text="Download complete!")
        time.sleep(1)
        progress_bar.empty()
        return True
    except Exception as e:
        progress_bar.error(f"Error: {str(e)}")
        return False

def get_ytid(input_url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'youtu\.be\/([a-zA-Z0-9_-]+)',  # Short URLs
        r'youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',  # Regular watch URLs
        r'youtube\.com\/shorts\/([a-zA-Z0-9_-]+)'  # Shorts URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_url)
        if match:
            return match.group(1)
    return None

def find_wav_files(directory):
    """Find WAV files in directory"""
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

@st.cache_resource
def transcribe_audio(wave_file):
    """Transcribe audio file using AssemblyAI"""
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(wave_file)
        return transcript.text if transcript else None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def convert_notebook_to_markdown(notebook_json):
    """Convert Jupyter notebook to markdown format"""
    try:
        notebook = nbformat.reads(json.dumps(notebook_json), as_version=4)
        markdown_exporter = MarkdownExporter()
        markdown_content, _ = markdown_exporter.from_notebook_node(notebook)
        return markdown_content
    except Exception as e:
        st.error(f"Error converting notebook to markdown: {str(e)}")
        return None

def reset_callback():
    """Reset all application state variables"""
    st.session_state.notebook_state = {
        'content': None,
        'filename': None
    }
    st.session_state.show_error = False
    st.session_state.error_message = ""
    st.session_state.markdown_content = None
    st.session_state.transcript_content = None
    st.session_state.generated_blog = None
    st.session_state.html_content = None
    st.session_state.zip_data = None
    st.session_state.submitted = False
    st.session_state.github_url = ""
    st.session_state.youtube_url = ""
    
    # Clean up any WAV files
    wav_files = find_wav_files(os.getcwd())
    for file in wav_files:
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not remove temporary audio file: {str(e)}")

def convert_markdown_to_html(markdown_text):
    """Convert markdown text to HTML using the markdown package"""
    return markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br']).convert(markdown_text)

def create_files_and_zip():
    """Creates markdown and HTML files from session state content and returns zipped bytes."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if 'generated_blog' in st.session_state:
            zip_file.writestr('blog.md', st.session_state.generated_blog)
        
        if 'html_content' in st.session_state:
            zip_file.writestr('blog.html', st.session_state.html_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def handle_download():
    """Callback function to handle the download button click."""
    st.session_state.zip_data = create_files_and_zip()
    reset_callback()

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Blog",
    page_icon="✍️",
    layout="wide"
)

with st.sidebar:
    # Add title and description
    st.title("✍️ Write Blog")
    st.warning(
        "Transform your Jupyter notebooks into polished technical blog posts. "
        "Add context from YouTube videos to enrich your content."
    )
    
    # Input method selection for notebook
    st.subheader("Jupyter Notebook")
    notebook_input_method = st.radio(
        "Choose notebook input method",
        ["Upload File", "GitHub URL"],
        key="notebook_input_method",
        help="Select how you want to provide your notebook"
    )
    
    if notebook_input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your notebook",
            type=['ipynb'],
            help="Upload a Jupyter notebook file"
        )
        if uploaded_file is not None:
            try:
                notebook_json = json.loads(uploaded_file.getvalue())
                st.session_state.notebook_state['content'] = json.dumps(notebook_json)
                st.session_state.notebook_state['filename'] = uploaded_file.name
                markdown_content = convert_notebook_to_markdown(notebook_json)
                st.session_state.markdown_content = preprocess_markdown_content(markdown_content)
            except Exception as e:
                st.error(f"Error processing notebook: {str(e)}")
    else:
        github_url = st.text_input(
            "Enter GitHub Notebook URL",
            key="github_url",
            on_change=on_url_change,
            placeholder="https://github.com/username/repo/blob/main/notebook.ipynb"
        )

    # Content input section
    st.subheader("Additional Content (Optional)")
    content_input_method = st.radio(
        "Choose content input method",
        ["YouTube URL", "Upload File"],
        key="content_input_method",
        help="Select how you want to provide additional content"
    )
    
    if content_input_method == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL",
            key="youtube_url",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        if youtube_url and is_valid_youtube_url(youtube_url):
            progress_bar = st.progress(0)
            if download_audio(youtube_url, progress_bar):
                wav_files = find_wav_files(os.getcwd())
                if wav_files:
                    with st.spinner("📝 Transcribing audio... This may take a few minutes..."):
                        transcript = transcribe_audio(wav_files[0])
                        if transcript:
                            st.session_state.transcript_content = transcript
                            try:
                                os.remove(wav_files[0])
                            except Exception as e:
                                st.warning(f"Could not remove temporary audio file: {str(e)}")
        elif youtube_url:
            st.error("Please enter a valid YouTube URL")
    else:
        uploaded_content = st.file_uploader(
            "Upload additional content",
            type=['txt', 'md'],
            help="Upload a text or markdown file with additional content"
        )
        if uploaded_content is not None:
            content = uploaded_content.getvalue().decode('utf-8')
            # Apply preprocessing to clean the content
            cleaned_content = preprocess_markdown_content(content)
            st.session_state.transcript_content = cleaned_content

    st.subheader("⚙️ Settings")
    llm_model = st.selectbox(
        "Select a model",
        ("claude-3-5-sonnet-20241022", "o1-mini", "gpt-4-turbo")
    )

    # Show both buttons but control their disabled state
    can_generate = st.session_state.get('markdown_content') is not None
    has_input = (bool(st.session_state.notebook_state['content']) or 
                bool(st.session_state.youtube_url.strip()))

    st.button(
        "Generate Blog",
        type="primary",
        on_click=submit_callback,
        disabled=not can_generate,
        use_container_width=True
    )

    st.button(
        "Reset All",
        type="primary",
        on_click=reset_callback,
        disabled=not has_input,
        use_container_width=True,
        help="Clear all inputs and generated content"
    )

# Show error message if there is one
if st.session_state.show_error:
    st.error(st.session_state.error_message)

# Create three columns for displaying content
if st.session_state.notebook_state['content'] is not None or st.session_state.transcript_content is not None:
    col1, col2, col3 = st.columns(3)

    # Column 1: Notebook Content
    with col1:
        if st.session_state.notebook_state['content'] is not None:
            st.subheader("Notebook IPYNB")
            notebook_json = json.loads(st.session_state.notebook_state['content'])
            
            st.text_area(
                "Notebook",
                json.dumps(notebook_json, indent=2),
                height=400
            )
            
            st.download_button(
                label="Download .ipynb file",
                data=st.session_state.notebook_state['content'],
                file_name=st.session_state.notebook_state['filename'],
                mime="application/x-ipynb+json",
                key="download_notebook"
            )

    # Column 2: Markdown Content
    with col2:
        if st.session_state.markdown_content:
            st.subheader("Notebook Markdown")
            st.text_area(
                "Preview",
                st.session_state.markdown_content,
                height=400
            )
            
            st.download_button(
                label="Download Markdown",
                data=st.session_state.markdown_content,
                file_name=f"{os.path.splitext(st.session_state.notebook_state['filename'])[0]}.md",
                mime="text/markdown",
                key="download_markdown"
            )

    # Column 3: Transcript Content
    with col3:
        if st.session_state.transcript_content:
            st.subheader("Video Transcript")
            st.text_area(
                "Transcript",
                st.session_state.transcript_content,
                height=400
            )
            
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript_content,
                file_name="transcript.txt",
                mime="text/plain",
                key="download_transcript"
            )

if not st.session_state.notebook_state['content'] and not st.session_state.transcript_content:
    if notebook_input_method == "Upload File":
        st.info("Please upload your notebook in the sidebar!", icon="👈")
    else:
        st.info("Please enter a GitHub URL in the sidebar!", icon="👈")

# Only process the blog generation if the submit button has been clicked
if (st.session_state.get('markdown_content') is not None and 
    st.session_state.submitted):
    
    system_prompt = """
    You are an experienced and prolific technical blog writer.
    """

    # Construct base prompt without transcript content
    base_prompt = f"""
        Create a technical blog post that integrates the following:
        
        Content Integration:
        - Make sure to preserve all code snippets exactly as provided
        - If mentioning or referring to this article being written please use "in this blog"
        - Remove emojis from title
        - Title format: "How to [Topic]"
        - After an abbreviation has been mentioned for the first time, in its
          second usage there is no need to abbreviate
        
        Structure and Style:
        1. Blog organization:
           - Introduction consists 5 sentences broken down into 3 paragraphs:
             In paragraph 1, write 1 sentence on the use case
             (start with a provocative question) and 1 sentence on problem statement. 
             In paragraph 2, write 1 sentence on the solution. 
             In paragraph 3, write 1 sentence on what is presented in the blog, 
             and 1 sentence on what readers will learn in this blog (please use bullet points 
             if possible)
           - Building the solution (show code along with concise high-level explanation,
             please title section based on what is being built, e.g. Building the [Solution]).
             Before proceeding to each sub-section, please write 1 sentence that provides a high-level look 
           - Conclusion (a recap of what was covered, what the reader built and how they can 
             adapt and improvise for their own use case in the future)
           - Resources section will list the links to resources that is provided in the input Markdown Content.
        
        2. Writing approach:
           - Professional yet accessible tone
           - Active voice
           - Direct reader address
           - Concise introduction focusing on value proposition

        Deliver the final output directly without meta-commentary.
        
        Input:
        Markdown Content: {st.session_state.markdown_content}
        """

    # Add transcript content only if available
    if st.session_state.get('transcript_content') is not None:
        base_prompt += f"\nVideo Transcript: {st.session_state.transcript_content}"
        base_prompt = base_prompt.replace("Content Integration:", 
                                        "Content Integration:\n        - Merge video transcript content into the markdown structure at relevant points")

    user_prompt = base_prompt

    st.subheader("Generated Blog")
    
    # Create a progress bar placeholder
    progress_bar = st.empty()
    progress_bar.progress(0, text="Starting blog generation...")
    
    try:
        # Simulate progress while waiting for API response
        for percent in range(0, 90, 10):
            time.sleep(0.1)  # Add small delay
            progress_bar.progress(percent, text=f"Generating blog content... {percent}%")

        if llm_model == "o1-mini":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            blog_content = completion.choices[0].message.content

        elif llm_model == "gpt-4-turbo":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            blog_content = completion.choices[0].message.content

        elif llm_model == "claude-3-5-sonnet-20241022":
            client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
            completion = client.messages.create(
                model=llm_model,
                max_tokens=4000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            blog_content = completion.content[0].text
        
        # Update to almost complete
        progress_bar.progress(90, text="Processing final output...")
        
        # Store the generated blog in session state
        st.session_state.generated_blog = blog_content
        st.session_state.html_content = convert_markdown_to_html(st.session_state.generated_blog)
        
        # Show completion
        progress_bar.progress(100, text="Blog generation complete!")
        time.sleep(0.5)  # Short delay before clearing
        progress_bar.empty()  # Clear the progress bar
        
        # Display the blog content
        with st.expander('See generated blog'):
            st.markdown(st.session_state.generated_blog)
                        
        st.write("**Markdown**")
        with st.expander("Generated Blog (Markdown)"):
            st.code(st.session_state.generated_blog, language='markdown')

        st.write("**HTML**")
        with st.expander("Generated Blog (HTML)"):
            st.code(st.session_state.html_content, language='html')

        st.download_button(
            label="📥 Download ZIP file",
            data=st.session_state.zip_data if st.session_state.zip_data else create_files_and_zip(),
            file_name="content.zip",
            mime="application/zip",
            key='download_button',
            help="Download the Markdown and HTML files as a zip",
            on_click=handle_download
        )

        # Reset the submitted state after generation is complete
        st.session_state.submitted = False
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error generating blog: {str(e)}")
