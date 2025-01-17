# Import statements and license headers
import streamlit as st
import requests
import os 
import json
import anthropic
import zipfile
import io
import time
import re
import nbformat
from nbconvert import MarkdownExporter
from urllib.parse import urlparse, parse_qs
from openai import OpenAI
from collections import defaultdict
import yt_dlp
import assemblyai as aai
from bs4 import BeautifulSoup

# Define valid categories at module level
VALID_CATEGORIES = {
    'getting-started': {
        'keywords': ['introduction', 'basics', 'fundamental', 'begin', 'start', 'first', 'new', 'learn', 
                    'tutorial', 'guide', 'quickstart', 'getting started', 'prerequisites'],
        'weight': 1.0
    },
    'data-engineering': {
        'keywords': ['etl', 'pipeline', 'data flow', 'integration', 'transformation', 
                    'data engineering', 'data pipeline', 'data integration', 'snowpark',
                    'data warehouse', 'schema', 'streaming'],
        'weight': 1.2
    },
    'cybersecurity': {
        'keywords': ['security', 'mfa', 'multi-factor authentication', 'authentication', 'authorization', 
                    'compliance', 'encrypt', 'credential', 'audit', 'access control',
                    'identity', 'privacy'],
        'weight': 1.2
    },
    'audit': {
        'keywords': ['audit', 'compliance', 'monitor', 'tracking', 'verification',
                    'validation', 'check', 'review', 'assessment'],
        'weight': 1.2
    },
    'streamlit': {
        'keywords': ['streamlit', 'st.', 'interactive dashboard', 'web interface',
                    'streamlit app', 'st.button', 'st.dataframe', 'st.header'],
        'weight': 1.3
    },
    'notebooks': {
        'keywords': ['notebook', 'jupyter', 'snowflake notebook', 'interactive notebook',
                    'notebook cell', 'code cell'],
        'weight': 1.1
    },
    'snowflake': {
        'keywords': ['snowflake', 'snowpark', 'warehouse', 'snowflake native', 
                    'snowflake integration', 'snowflake table'],
        'weight': 1.2
    },
    'featured': {
        'keywords': [],
        'weight': 1.0
    }
}

# Initialize session state variables
def init_session_state():
    if 'blog_content' not in st.session_state:
        st.session_state.blog_content = None
    if 'generated_blog' not in st.session_state:
        st.session_state.generated_blog = None
    if 'zip_data' not in st.session_state:
        st.session_state.zip_data = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'custom_title' not in st.session_state:
        st.session_state.custom_title = None
    if 'author_name' not in st.session_state:
        st.session_state.author_name = None
    if 'github_url' not in st.session_state:
        st.session_state.github_url = ""
    if 'youtube_url' not in st.session_state:
        st.session_state.youtube_url = ""
    if 'transcript_content' not in st.session_state:
        st.session_state.transcript_content = None
    if 'show_error' not in st.session_state:
        st.session_state.show_error = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = None
    if 'previous_input_method' not in st.session_state:
        st.session_state.previous_input_method = "Upload Markdown File"

# GitHub-related functions
def extract_github_data(url):
    """Extract repository data from GitHub URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', attrs={
            'type': 'application/json',
            'data-target': 'react-app.embeddedData'
        })
        
        if script_tag:
            try:
                return json.loads(script_tag.string)
            except json.JSONDecodeError:
                return None
    return None

def process_github_content(github_url):
    """Process content from GitHub URL, handling both direct file links and repository URLs"""
    if not is_valid_github_url(github_url):
        return None, "Invalid GitHub URL format"
        
    try:
        # For repository URLs (not direct file links)
        if not (github_url.endswith('.ipynb') or github_url.endswith('.md')):
            json_data = extract_github_data(github_url)
            if not json_data:
                return None, "Failed to extract repository data"
                
            # Parse repository information
            repo_owner = json_data['payload']['repo']['ownerLogin']
            repo_name = json_data['payload']['repo']['name']
            branch = json_data['payload']['repo']['defaultBranch']
            
            # Find suitable files
            tree_items = json_data['payload']['tree']['items']
            target_files = [item['path'] for item in tree_items 
                          if item['path'].endswith(('.ipynb', '.md'))]
            
            if not target_files:
                return None, "No suitable .ipynb or .md file found in repository"
                
            # Prefer .ipynb files over .md files
            target_file = next((f for f in target_files if f.endswith('.ipynb')), target_files[0])
            raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/refs/heads/{branch}/{target_file}"
        else:
            # Handle direct file links
            parsed = urlparse(github_url)
            path_parts = parsed.path.split('/')
            
            if 'blob' in path_parts:
                blob_index = path_parts.index('blob')
                path_parts.pop(blob_index)
                path_parts.insert(blob_index, 'refs/heads')
            
            raw_url = f"https://raw.githubusercontent.com{'/'.join(path_parts)}"
        
        # Fetch and process content
        response = requests.get(raw_url)
        response.raise_for_status()
        
        if raw_url.endswith('.ipynb'):
            notebook_json = json.loads(response.content)
            markdown_exporter = MarkdownExporter()
            content, _ = markdown_exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_json), as_version=4))
            return content, None
        else:
            return response.content.decode('utf-8'), None
            
    except Exception as e:
        return None, f"Error processing GitHub content: {str(e)}"

def is_valid_github_url(url):
    """Validate GitHub URL for both direct links and repository URLs"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        if parsed.netloc != "github.com":
            return False
            
        path_parts = parsed.path.split('/')
        if len(path_parts) < 3:  # Need at least owner/repo
            return False
            
        # Check for direct file link
        if path_parts[-1].endswith(('.ipynb', '.md')):
            return True
            
        # Check for repository URL
        return len(path_parts) >= 4 and path_parts[3] in ['tree', 'blob']
        
    except:
        return False

# YouTube-related functions
def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube or YouTube Shorts URL"""
    patterns = [
        r'https?:\/\/(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)(?:\?.*)?$',
        r'https?:\/\/(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(?:&.*)?$',
        r'https?:\/\/(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)(?:\?.*)?$'
    ]
    
    for pattern in patterns:
        if re.match(pattern, url.strip()):
            return True
    return False

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

# Content processing functions
def update_content_state():
    """Update the application state with new content"""
    st.session_state.show_error = False
    st.session_state.error_message = ""
    st.session_state.blog_content = None
    
    if st.session_state.github_url:
        content, error = process_github_content(st.session_state.github_url)
        if error:
            st.session_state.error_message = error
            st.session_state.show_error = True
        else:
            st.session_state.blog_content = content
            
            # Update categories based on new content
            if content:
                suggested_categories = identify_categories(content)
                if 'selected_categories' not in st.session_state or not st.session_state.selected_categories:
                    st.session_state.selected_categories = suggested_categories

def on_url_change():
    """Handle GitHub URL changes"""
    update_content_state()

def has_input_content():
    """Check if there is any input content"""
    return (st.session_state.blog_content is not None or 
            bool(st.session_state.github_url.strip()) or 
            'uploaded_file' in st.session_state)

def identify_categories(content):
    """Identify categories based on the content"""
    content_lower = content.lower()
    scores = defaultdict(float)
    
    for category, data in VALID_CATEGORIES.items():
        matches = 0
        for keyword in data['keywords']:
            if ' ' not in keyword:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, content_lower):
                matches += 1
        
        if matches > 0:
            scores[category] = matches * data['weight']
    
    matched_categories = []
    if scores:
        max_score = max(scores.values())
        threshold = max_score * 0.3
        matched_categories = [cat for cat, score in scores.items() 
                            if score >= threshold]
        
        if len(matched_categories) >= 2 and 'featured' in VALID_CATEGORIES:
            matched_categories.append('featured')
    
    if not matched_categories:
        matched_categories = ['getting-started']
    
    return matched_categories

# File handling functions
def extract_title(content):
    """Extract the article title from the generated content or use custom title"""
    if st.session_state.custom_title:
        clean_title = re.sub(r'[^\w\s-]', '', st.session_state.custom_title)
        clean_title = clean_title.replace(' ', '_').lower()
        return clean_title
        
    try:
        match = re.search(r'# (.+?)(?=\n|\r)', content)
        if match:
            title = match.group(1).strip()
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = clean_title.replace(' ', '_').lower()
            return clean_title
    except Exception:
        pass
    return 'tutorial'

def create_zip():
    """Creates a zip file with the markdown file and assets folder"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if 'generated_blog' in st.session_state:
            title = extract_title(st.session_state.generated_blog)
            
            # Add source information if from GitHub
            if st.session_state.github_url:
                source_info = f"""
                Source: {st.session_state.github_url}
                Retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}
                """
                zip_file.writestr(f"{title}/source_info.txt", source_info)
            
            # Add main content
            file_path = f"{title}/{title}.md"
            zip_file.writestr(file_path, st.session_state.generated_blog)
            
            # Add assets folder
            assets_path = f"{title}/assets/.gitkeep"
            zip_file.writestr(assets_path, "")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def handle_download():
    """Callback function to handle the download button click"""
    st.session_state.zip_data = create_zip()
    reset_callback()

def reset_callback():
    """Reset all application state variables"""
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize necessary session state variables
    st.session_state.blog_content = None
    st.session_state.generated_blog = None
    st.session_state.zip_data = None
    st.session_state.submitted = False
    st.session_state.custom_title = None
    st.session_state.author_name = None
    st.session_state.github_url = ""
    st.session_state.youtube_url = ""
    st.session_state.transcript_content = None
    st.session_state.show_error = False
    st.session_state.error_message = ""
    st.session_state.selected_categories = None
    st.session_state.previous_input_method = "Upload Markdown File"
    
    # Clean up any WAV files
    wav_files = find_wav_files(os.getcwd())
    for file in wav_files:
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not remove temporary audio file: {str(e)}")
            
    # Force a rerun to clear the interface
    st.rerun()

def create_content_preview():
    """Create a preview of the content"""
    if st.session_state.blog_content:
        preview_length = 200
        preview = st.session_state.blog_content[:preview_length]
        if len(st.session_state.blog_content) > preview_length:
            preview += "..."
        return preview
    return None

def get_user_prompt(blog_content, transcript_content=None):
    """Construct the user prompt for the LLM"""
    prompt = f"""
    Create a technical tutorial by integrating the following:
    
    Content Integration:
    - Make sure to preserve all code snippets exactly as provided
    - After an abbreviation has been mentioned for the first time, in its
      second usage there is no need to abbreviate
    
    Structure and Style:
    1. Tutorial organization:
       - Introduction consists of a clear overview of what will be accomplished
       - Step-by-step instructions with clear headings
       - Code blocks properly formatted and explained
       - Conclusion with recap and next steps
       
    2. Writing approach:
       - Professional yet accessible tone
       - Active voice
       - Direct reader address
       - Clear, numbered steps

    Input Content: {blog_content}
    """
    
    if transcript_content:
        prompt += f"\nVideo Transcript: {transcript_content}"
    
    if st.session_state.author_name:
        prompt += f"\nAuthor: {st.session_state.author_name}"
    
    if st.session_state.custom_title:
        prompt += f"\nTitle: {st.session_state.custom_title}"
    
    if st.session_state.selected_categories:
        prompt += f"\nCategories: {', '.join(st.session_state.selected_categories)}"
    
    return prompt


# Verify API keys
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("Please set your OpenAI API key in Streamlit secrets as OPENAI_API_KEY")
    st.stop()
if 'ANTHROPIC_API_KEY' not in st.secrets:
    st.error("Please set your Anthropic API key in Streamlit secrets as ANTHROPIC_API_KEY")
    st.stop()
if 'AAI_KEY' not in st.secrets:
    st.error("Please set your AssemblyAI API key in Streamlit secrets as AAI_KEY")
    st.stop()

# Initialize AssemblyAI client
aai.settings.api_key = st.secrets['AAI_KEY']

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Quickstarts",
    page_icon="⏩",
    layout="wide"
)

# Initialize session state
init_session_state()

# Main UI
with st.sidebar:
    st.title("⏩ Write Quickstarts")
    st.warning(
        "Transform your technical content into Quick Start tutorials."
    )
    
    st.subheader("📃 Input Content")
    input_method = st.radio(
        "Choose input method",
        ["GitHub URL", "Upload Markdown File"],
        help="Select how you want to provide your content"
    )
    
    # Clear content if input method changes
    if input_method != st.session_state.previous_input_method:
        st.session_state.blog_content = None
        st.session_state.github_url = ""
        if 'uploaded_file' in st.session_state:
            del st.session_state.uploaded_file
        st.session_state.previous_input_method = input_method
        st.rerun()
    
    if input_method == "GitHub URL":
        github_url = st.text_input(
            "Enter GitHub URL",
            key="github_url",
            on_change=on_url_change,
            placeholder="https://github.com/username/repo/blob/main/file.{md,ipynb} or repository URL"
        )
        
        st.caption("""
        You can provide either:
        - Direct link to a .ipynb or .md file
        - Repository URL (will automatically find relevant files)
        """)
    else:
        uploaded_file = st.file_uploader(
            "Upload your content",
            type=['md', 'txt', 'ipynb'],
            help="Upload a markdown file or notebook",
            key="uploaded_file"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.ipynb'):
                    notebook_json = json.loads(uploaded_file.getvalue().decode('utf-8'))
                    markdown_exporter = MarkdownExporter()
                    content, _ = markdown_exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_json), as_version=4))
                    st.session_state.blog_content = content
                else:
                    st.session_state.blog_content = uploaded_file.getvalue().decode('utf-8')
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                st.session_state.blog_content = None

    # YouTube Video section
    st.subheader("📺 YouTube Video (Optional)")
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

    # Settings section
    st.subheader("⚙️ Settings")
    llm_model = st.selectbox(
        "Select a model",
        ("claude-3-5-sonnet-20241022", "o1-mini", "gpt-4-turbo")
    )
    
    use_custom_title = st.checkbox('Specify Quickstart Title')
    if use_custom_title:
        st.session_state.custom_title = st.text_input(
            'Enter Quickstart Title',
            value=st.session_state.custom_title if st.session_state.custom_title else '',
            help="This will be used to name the output folder and ZIP file"
        )
    
    use_custom_author = st.checkbox('Specify Author Name')
    if use_custom_author:
        st.session_state.author_name = st.text_input(
            'Enter Author Name',
            value=st.session_state.author_name if st.session_state.author_name else '',
            help="This will be used in the Author field of the Quickstart template"
        )

    # Categories section
    if st.session_state.blog_content:
        st.subheader("📑 Categories")
        suggested_categories = identify_categories(st.session_state.blog_content)
        st.session_state.selected_categories = st.multiselect(
            "Select categories",
            options=list(VALID_CATEGORIES.keys()),
            default=suggested_categories,
            help="Choose categories that best match the content"
        )

    # Buttons
    st.button(
        "Generate Quickstart",
        type="primary",
        on_click=lambda: setattr(st.session_state, 'submitted', True),
        disabled=st.session_state.blog_content is None,
        use_container_width=True
    )
    
    st.button(
        "Reset All",
        type="secondary",
        on_click=reset_callback,
        disabled=not has_input_content(),
        use_container_width=True
    )

# Show error message if there is one
if st.session_state.show_error:
    st.error(st.session_state.error_message)

# Main content area
if not st.session_state.blog_content:
    if input_method == "GitHub URL":
        st.info("Enter a GitHub URL in the sidebar. You can use either a direct file link or a repository URL!", icon="👈")
    else:
        st.info("Please upload your content file in the sidebar!", icon="👈")
else:
    # Display content preview
    if st.session_state.transcript_content:
        col1, col2 = st.columns(2)
    else:
        col1, = st.columns(1)
    
    with col1:
        st.subheader("Content Preview")
        if input_method == "GitHub URL":
            st.caption(f"Source: {st.session_state.github_url}")
        
        st.text_area(
            "Content",
            st.session_state.blog_content,
            height=400
        )
        
        # Download button for input content
        filename = os.path.basename(st.session_state.github_url) if input_method == "GitHub URL" else "content.md"
        st.download_button(
            label="Download Input Content",
            data=st.session_state.blog_content,
            file_name=filename,
            mime="text/markdown" if filename.endswith('.md') else "application/x-ipynb+json",
            key="download_input"
        )

    # Display transcript if available
    if st.session_state.transcript_content:
        with col2:
            st.subheader("Video Transcript")
            st.text_area(
                "Preview",
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

# Generate quickstart content
if st.session_state.blog_content is not None and st.session_state.submitted:
    st.subheader("Generated Quickstarts")
    
    progress_bar = st.empty()
    progress_bar.progress(0, text="Starting quickstart generation...")
    
    try:
        for percent in range(0, 90, 10):
            time.sleep(0.1)
            progress_bar.progress(percent, text=f"Generating Quickstarts Content... {percent}%")
        
        # Get prompts
        system_prompt = """
        You are an experienced technical writer specializing in creating clear, 
        structured tutorials from existing technical content.
        """
        user_prompt = get_user_prompt(st.session_state.blog_content, st.session_state.transcript_content)
        
        # Generate content based on selected model
        if llm_model == "o1-mini":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            tutorial_content = completion.choices[0].message.content

        elif llm_model == "gpt-4-turbo":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            tutorial_content = completion.choices[0].message.content

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
            tutorial_content = completion.content[0].text
        
        progress_bar.progress(90, text="Processing final output...")
        
        # Store generated content
        st.session_state.generated_blog = tutorial_content
        
        progress_bar.progress(100, text="Quickstarts generation complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        # Display generated content
        with st.expander("See Generated Quickstarts", expanded=True):
            st.code(st.session_state.generated_blog, language='markdown')

        # Download button for zip file
        st.download_button(
            label="📥 Download ZIP",
            data=st.session_state.zip_data if st.session_state.zip_data else create_zip(),
            file_name=f"{extract_title(st.session_state.generated_blog)}.zip",
            mime="application/zip",
            key='download_button',
            help="Download the Quickstarts with assets folder",
            on_click=handle_download
        )
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error Generating Quickstarts: {str(e)}")

    # Reset the submitted state
    st.session_state.submitted = False
