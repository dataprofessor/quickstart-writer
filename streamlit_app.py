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

# Template for the quickstart guide
quickstart_template = '''
author: [Your Name]
id: [unique-identifier-with-dash]
summary: [One to two sentences describing what this guide covers]
categories: [comma-separated list: e.g., featured, getting-started, data-engineering]
environments: web
status: Published
feedback link: https://github.com/Snowflake-Labs/sfguides/issues
tags: [Comma-separated list of relevant technologies and concepts]

# [Article Title]
<!-- ------------------------ -->
## Overview
Duration: [Minutes as an integer only]

[One to two paragraphs introducing the topic and what will be accomplished]

### What You'll Learn
- [Key learning objective 1]
- [Key learning objective 2]
- [Key learning objective 3]
- [Add more as needed]

### What You'll Build
[Describe the end result/application/solution the reader will create]

[Optional: Include screenshot or diagram of final result]

### Prerequisites
- [Required account/access/subscription]
- [Required software/tools]
- [Required knowledge/skills]
- [Other requirements]

<!-- ------------------------ -->
## Setup
Duration: [X]

### [Setup Step 1 - e.g., Environment Configuration]
[Detailed instructions]

```[language]
[code snippet if applicable]
```

### [Setup Step 2 - e.g., Installation]
[Detailed instructions]

> aside positive
> IMPORTANT:
> - [Critical note 1]
> - [Critical note 2]

<!-- ------------------------ -->
## [Main Content Section 1]
Duration: [X]

### [Subsection 1.1]
[Detailed explanation]

```[language]
[code snippet if applicable]
```

### [Subsection 1.2]
[Detailed explanation]

[Include screenshots/diagrams where helpful]

<!-- ------------------------ -->
## [Main Content Section 2]
Duration: [X]

[Repeat structure as needed for additional main sections]

## Conclusion and Resources
Duration: [X]

### What You Learned
- [Key takeaway 1]
- [Key takeaway 2]
- [Key takeaway 3]

### Related Resources

Articles:
- [Resource link 1 with description]
- [Resource link 2 with description]
- [Resource link 3 with description]

Documentation:
- [Relevant documentation link 1]
- [Relevant documentation link 2]

Additional Reading:
- [Blog/article link 1]
- [Blog/article link 2]
'''

# Initialize session state variables
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
if 'input_method' not in st.session_state:
    st.session_state.input_method = "GitHub URL of Notebook"
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "claude-3-5-sonnet-20241022"
if 'use_custom_title' not in st.session_state:
    st.session_state.use_custom_title = False
if 'use_custom_author' not in st.session_state:
    st.session_state.use_custom_author = False
if 'is_youtube_processing' not in st.session_state:
    st.session_state.is_youtube_processing = False
if 'previous_youtube_url' not in st.session_state:
    st.session_state.previous_youtube_url = ""

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

def is_valid_github_url(url):
    """Validate GitHub URL for repository or direct file links"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        
        # Basic GitHub URL validation
        if parsed.netloc != "github.com" or len(path_parts) < 3:
            return False
            
        # Case 1: Direct file link validation
        if "blob" in path_parts:
            return path_parts[-1].endswith('.ipynb') or path_parts[-1].endswith('.md')
            
        # Case 2: Repository link validation
        if "tree" in path_parts:
            return True
            
        return False
    except:
        return False

def get_raw_github_url(github_url, json_data=None):
    """Convert GitHub URL to raw content URL"""
    parsed = urlparse(github_url)
    path_parts = parsed.path.split('/')
    
    # Case 1: Direct file link
    if "blob" in path_parts:
        # Get the index of 'blob' and extract relevant parts
        blob_index = path_parts.index('blob')
        owner = path_parts[1]
        repo = path_parts[2]
        branch = path_parts[blob_index + 1]
        file_path = '/'.join(path_parts[blob_index + 2:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    
    # Case 2: Repository link
    elif "tree" in path_parts and json_data:
        repo_owner = json_data['payload']['repo']['ownerLogin']
        repo_name = json_data['payload']['repo']['name']
        branch = json_data['payload']['repo']['defaultBranch']
        
        # Find the .ipynb file path
        tree_items = json_data['payload']['tree']['items']
        ipynb_path = None
        for item in tree_items:
            if item['path'].endswith('.ipynb'):
                ipynb_path = item['path']
                break
                
        if ipynb_path:
            return f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{ipynb_path}"
    
    return None

def extract_github_data(url):
    """Extract repository data from GitHub page"""
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
                return "Error: Could not parse JSON data"
        else:
            return "Error: Script tag not found"
    else:
        return f"Error: Failed to fetch page (Status code: {response.status_code})"

def get_file_content(github_url):
    """Fetch content from GitHub URL"""
    try:
        # Case 1: Direct file link
        if "blob" in github_url:
            raw_url = get_raw_github_url(github_url)
            
        # Case 2: Repository link
        elif "tree" in github_url:
            json_data = extract_github_data(github_url)
            if isinstance(json_data, str) and "Error" in json_data:
                raise Exception(json_data)
            raw_url = get_raw_github_url(github_url, json_data)
            
        if not raw_url:
            raise Exception("Could not generate raw URL")
            
        response = requests.get(raw_url)
        response.raise_for_status()
        
        filename = os.path.basename(raw_url)
        if filename.endswith('.ipynb'):
            notebook_json = json.loads(response.content)
            markdown_exporter = MarkdownExporter()
            content, _ = markdown_exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_json), as_version=4))
            return content
        else:
            return response.content.decode('utf-8')
            
    except Exception as e:
        st.session_state.error_message = f"Error: {str(e)}"
        st.session_state.show_error = True
        return None

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

def identify_categories(content):
    """Identify categories based on the content."""
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

def extract_title(content):
    """Extract the article title from the generated content or use custom title."""
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
    """Creates a zip file with the markdown file and assets folder inside a tutorial folder."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if 'generated_blog' in st.session_state:
            title = extract_title(st.session_state.generated_blog)
            file_path = f"{title}/{title}.md"
            zip_file.writestr(file_path, st.session_state.generated_blog)
            assets_path = f"{title}/assets/.gitkeep"
            zip_file.writestr(assets_path, "")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def handle_download():
    """Callback function to handle the download button click."""
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
    st.session_state.input_method = "GitHub URL of Notebook"
    st.session_state.llm_model = "claude-3-5-sonnet-20241022"
    st.session_state.use_custom_title = False
    st.session_state.use_custom_author = False
    st.session_state.is_youtube_processing = False
    st.session_state.previous_youtube_url = ""
    
    # Clean up any WAV files
    wav_files = find_wav_files(os.getcwd())
    for file in wav_files:
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not remove temporary audio file: {str(e)}")
            
    # Force a rerun to clear the interface
    st.rerun()

def has_input_content():
    """Check if there is any input content"""
    return (st.session_state.blog_content is not None or 
            bool(st.session_state.github_url.strip()) or 
            'uploaded_file' in st.session_state)

def submit_callback():
    """Callback function to handle the submit button click"""
    st.session_state.submitted = True

def on_input_method_change():
    """Handle input method changes"""
    if st.session_state.input_method != st.session_state.previous_input_method:
        st.session_state.blog_content = None
        st.session_state.github_url = ""
        if 'uploaded_file' in st.session_state:
            del st.session_state.uploaded_file
        st.session_state.previous_input_method = st.session_state.input_method

def on_youtube_url_change():
    """Handle YouTube URL changes to prevent redundant processing"""
    new_url = st.session_state.youtube_url
    if (new_url and 
        not st.session_state.is_youtube_processing and 
        new_url != st.session_state.previous_youtube_url):
        
        st.session_state.previous_youtube_url = new_url
        st.session_state.is_youtube_processing = True
        st.session_state.transcript_content = None
        
        if is_valid_youtube_url(new_url):
            progress_bar = st.progress(0)
            if download_audio(new_url, progress_bar):
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
        
        st.session_state.is_youtube_processing = False

def on_url_change():
    """Handle GitHub URL changes"""
    st.session_state.show_error = False
    st.session_state.error_message = ""
    st.session_state.blog_content = None
    
    if st.session_state.github_url:
        if is_valid_github_url(st.session_state.github_url):
            content = get_file_content(st.session_state.github_url)
            if content:
                st.session_state.blog_content = content
        else:
            st.session_state.error_message = "Invalid GitHub URL. Please provide a valid GitHub URL pointing to a Jupyter notebook (.ipynb) or Markdown (.md) file, or a repository containing such files."
            st.session_state.show_error = True

def on_settings_change():
    """Handle settings changes"""
    # Update custom title visibility
    if st.session_state.use_custom_title != st.session_state.get('previous_use_custom_title', False):
        st.session_state.previous_use_custom_title = st.session_state.use_custom_title
        if not st.session_state.use_custom_title:
            st.session_state.custom_title = None
            
    # Update custom author visibility
    if st.session_state.use_custom_author != st.session_state.get('previous_use_custom_author', False):
        st.session_state.previous_use_custom_author = st.session_state.use_custom_author
        if not st.session_state.use_custom_author:
            st.session_state.author_name = None

def get_user_prompt(blog_content, transcript_content=None):
    """Construct the user prompt for the LLM"""
    prompt = f"""
    Create a technical tutorial by filling out the article template {quickstart_template}
    by integrating content and code from the attached blog content {blog_content}
    """
    
    if transcript_content:
        prompt += f"\nand incorporating relevant information from the video transcript: {transcript_content}"
    
    prompt += f"""
    {f'Please use "{st.session_state.author_name}" as the author name.' if st.session_state.author_name else ''}
    {f'Please use "{extract_title(st.session_state.generated_blog)}" as the id.' if st.session_state.custom_title else ''}
    {'Please use the following categories: ' + ', '.join(st.session_state.selected_categories) if st.session_state.selected_categories else ''}
            
    Writing approach:
    - Professional yet accessible tone
    - Active voice
    - Direct reader address
    - Concise introduction focusing on value proposition

    Notes:
    - If author name is not provided please add a placeholder like [First Name] [Last Name]
    - Please have the article title start with gerunds (Building, Performing, etc.)
    - In ## Setup section, please say something like: Firstly, to follow along with this quickstart, 
      you can download the [pre-made notebook]({st.session_state.github_url}) from the 
      `Snowflake-Labs`/`snowflake-demo-notebooks` GitHub repo.
    - If mentioning about installing Python packages. Please say something like the following but rephrase:
      Notebooks comes pre-installed with common Python libraries for data science and machine learning, 
      such as numpy, pandas, matplotlib, and more! If you are looking to use other packages, click on the 
      Packages dropdown on the top right to add additional packages to your notebook.
    - Please ensure that there is mention of the following in ## Overview: 
        ### What You'll Need
        Access to a [Snowflake account](https://signup.snowflake.com/)
    - In the Resources section, if you don't have the URL, please don't mention about it
      however if the URL is available in the provided blog please get it and use it
    - For the Duration, please give an estimate for reading and completing the task mentioned in each section.
    - In the Conclusion section, please start with a concluding remark that begins with 'Congratulations! 
      You've successfully' followed by 1-2 sentence summary of what was built in this tutorial. Please have
      this be the first paragraph of the Conclusion section prior to any sub-sections. For any closing remarks 
      like Happy Coding please make sure to have it as a normal text.
    - For the "What you Learned" section please write in past tense like built, created, analyzed, etc.
    - Make sure that the generated output don't have enclosing ``` symbols at its top-most and bottom-post.
    - Please see if you can include links from the provided input blog that starts with https://docs.snowflake.com/en
      to the 'Articles:' segment of the Conclusion section.
    - If provided blog contains mention of Streamlit please add [Streamlit Documentation](https://docs.streamlit.io/)
      to the 'Documention' segment of the Conclusion section.
    - Add [Snowflake Documentation](https://docs.snowflake.com/) to the 'Documention' segment of the Conclusion section.
    """
    
    return prompt

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Quickstarts",
    page_icon="⏩",
    layout="wide"
)

# Main UI implementation
with st.sidebar:
    st.title("⏩ Write Quickstarts")
    st.warning("Transform your technical content into Quick Start tutorials.")

    st.subheader("📃 Input Content")
    
    st.radio(
        "Choose input method",
        ["GitHub URL of Notebook", "Upload Markdown File"],
        key="input_method",
        on_change=on_input_method_change,
        help="Select how you want to provide your content"
    )
    
    if st.session_state.input_method == "Upload Markdown File":
        uploaded_file = st.file_uploader(
            "Upload your content (markdown or notebook file)",
            type=['md', 'txt', 'ipynb'],
            help="Upload a file containing your content",
            key="uploaded_file"
        )
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.ipynb'):
                notebook_json = json.loads(uploaded_file.getvalue().decode('utf-8'))
                markdown_exporter = MarkdownExporter()
                content, _ = markdown_exporter.from_notebook_node(nbformat.reads(json.dumps(notebook_json), as_version=4))
                st.session_state.blog_content = content
            else:
                st.session_state.blog_content = uploaded_file.getvalue().decode('utf-8')
    else:
        st.text_input(
            "Enter GitHub URL",
            key="github_url",
            on_change=on_url_change,
            placeholder="https://github.com/username/repo/blob/main/file.{md,ipynb}"
        )

    # YouTube Video section with state management
    st.subheader("📺 YouTube Video (Optional)")
    st.text_input(
        "Enter YouTube URL",
        key="youtube_url",
        on_change=on_youtube_url_change,
        placeholder="https://www.youtube.com/watch?v=..."
    )

    # Settings section with state management
    st.subheader("⚙️ Settings")
    st.selectbox(
        "Select a model",
        ("o1-mini", "gpt-4-turbo", "claude-3-5-sonnet-20241022"),
        key="llm_model"
    )
    
    st.checkbox(
        'Specify Quickstart Title',
        key="use_custom_title",
        on_change=on_settings_change
    )
    
    if st.session_state.use_custom_title:
        st.text_input(
            'Enter Quickstart Title',
            value=st.session_state.custom_title if st.session_state.custom_title else '',
            key="custom_title",
            help="This will be used to name the output folder and ZIP file"
        )
    
    st.checkbox(
        'Specify Author Name',
        key="use_custom_author",
        on_change=on_settings_change
    )
    
    if st.session_state.use_custom_author:
        st.text_input(
            'Enter Author Name',
            value=st.session_state.author_name if st.session_state.author_name else '',
            key="author_name",
            help="This will be used in the Author field of the Quickstart template"
        )

    # Categories section
    st.subheader("📑 Categories")
    if st.session_state.blog_content:
        suggested_categories = identify_categories(st.session_state.blog_content)
        st.session_state.selected_categories = st.multiselect(
            "Select categories",
            options=list(VALID_CATEGORIES.keys()),
            default=suggested_categories,
            help="Choose one or more categories that best match the content"
        )

    # Add Submit button - disabled if no blog content
    st.button(
        "Submit",
        type="primary",
        on_click=submit_callback,
        disabled=st.session_state.blog_content is None,
        use_container_width=True
    )
    
    # Add Reset button - enabled if there's any input content
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

if not st.session_state.blog_content:
    if st.session_state.input_method == "Upload Markdown File":
        st.info("Please upload your content file in the sidebar!", icon="👈")
    else:
        st.info("Please enter a GitHub URL in the sidebar!", icon="👈")
else:
    # Create columns for content display
    if st.session_state.transcript_content:
        col1, col2 = st.columns(2)
    else:
        col1, = st.columns(1)  # Using tuple unpacking for single column
    
    # Display input content
    with col1:
        # Dynamic header based on input method
        header_text = "Uploaded Markdown" if st.session_state.input_method == "Upload Markdown File" else "Notebook Markdown"
        st.subheader(header_text)
        st.text_area(
            "Preview",
            st.session_state.blog_content,
            height=400
        )
        
        filename = "content.md"
        if st.session_state.input_method == "GitHub URL of Notebook" and st.session_state.github_url:
            filename = os.path.basename(st.session_state.github_url)
        
        st.download_button(
            label="Download Input Content",
            data=st.session_state.blog_content,
            file_name=filename,
            mime="text/markdown" if filename.endswith('.md') else "text/plain",
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

# Only generate content if submitted
if st.session_state.blog_content is not None and st.session_state.submitted:
    system_prompt = """
    You are an experienced technical writer specializing in creating clear, 
    structured tutorials from existing technical content.
    """

    user_prompt = get_user_prompt(st.session_state.blog_content, st.session_state.transcript_content)

    st.subheader("Generated Quickstarts")
    
    # Create a progress bar placeholder
    progress_bar = st.empty()
    progress_bar.progress(0, text="Starting quickstart generation...")
    
    try:
        for percent in range(0, 90, 10):
            time.sleep(0.1)
            progress_bar.progress(percent, text=f"Generating Quickstarts Content... {percent}%")

        if st.session_state.llm_model == "o1-mini":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=st.session_state.llm_model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            tutorial_content = completion.choices[0].message.content

        elif st.session_state.llm_model == "gpt-4-turbo":
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            completion = client.chat.completions.create(
                model=st.session_state.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            tutorial_content = completion.choices[0].message.content

        elif st.session_state.llm_model == "claude-3-5-sonnet-20241022":
            client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
            completion = client.messages.create(
                model=st.session_state.llm_model,
                max_tokens=4000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            tutorial_content = completion.content[0].text
        
        progress_bar.progress(90, text="Processing final output...")
        
        # Store the generated tutorial
        st.session_state.generated_blog = tutorial_content
        
        progress_bar.progress(100, text="Quickstarts generation complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        # Display the generated tutorial
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
