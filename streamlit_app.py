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
from urllib.parse import urlparse
from openai import OpenAI
from collections import defaultdict

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
if 'show_error' not in st.session_state:
    st.session_state.show_error = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = None

# Verify API keys
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("Please set your OpenAI API key in Streamlit secrets as OPENAI_API_KEY")
    st.stop()
if 'ANTHROPIC_API_KEY' not in st.secrets:
    st.error("Please set your Anthropic API key in Streamlit secrets as ANTHROPIC_API_KEY")
    st.stop()

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

# GitHub URL handling functions
def is_valid_github_url(url):
    """Validate GitHub URL for Jupyter notebook or Markdown file"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        return (
            parsed.netloc == "github.com" and
            len(path_parts) >= 3 and
            ("blob" in parsed.path or "tree" in parsed.path) and
            (path_parts[-1].endswith('.ipynb') or path_parts[-1].endswith('.md'))
        )
    except:
        return False

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
            return content
        else:
            return response.content.decode('utf-8')
        
    except Exception as e:
        st.session_state.error_message = f"Error: {str(e)}"
        st.session_state.show_error = True
        return None

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
            st.session_state.error_message = "Invalid GitHub URL. Please provide a valid GitHub URL pointing to a Jupyter notebook (.ipynb) or Markdown (.md) file."
            st.session_state.show_error = True

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
    st.session_state.blog_content = None
    st.session_state.generated_blog = None
    st.session_state.zip_data = None
    st.session_state.submitted = False
    st.session_state.custom_title = None
    st.session_state.author_name = None
    st.session_state.github_url = ""
    st.session_state.show_error = False
    st.session_state.error_message = ""
    st.session_state.selected_categories = None

def submit_callback():
    """Callback function to handle the submit button click"""
    st.session_state.submitted = True

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Quickstarts",
    page_icon="⏩",
    layout="wide"
)

with st.sidebar:
    st.title("⏩ Write Quickstarts")
    st.warning(
        "Transform your technical content into Quick Start tutorials."
    )
    
    input_method = st.radio(
        "Choose input method",
        ["Upload File", "GitHub URL"],
        help="Select how you want to provide your content"
    )
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload your content (markdown or notebook file)",
            type=['md', 'txt', 'ipynb'],
            help="Upload a file containing your content"
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
        github_url = st.text_input(
            "Enter GitHub URL",
            key="github_url",
            on_change=on_url_change,
            placeholder="https://github.com/username/repo/blob/main/file.{md,ipynb}"
        )

    st.subheader("⚙️ Settings")
    llm_model = st.selectbox(
        "Select a model",
        ("o1-mini", "gpt-4-turbo", "claude-3-5-sonnet-20241022")
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
    
    # Add Reset button - disabled if content hasn't been generated yet
    st.button(
        "Reset All",
        type="primary",
        on_click=reset_callback,
        disabled=not st.session_state.submitted,
        use_container_width=True
    )

# Show error message if there is one
if st.session_state.show_error:
    st.error(st.session_state.error_message)

if not st.session_state.blog_content:
    if input_method == "Upload File":
        st.info("Please upload your content file in the sidebar!", icon="👈")
    else:
        st.info("Please enter a GitHub URL in the sidebar!", icon="👈")

# Only generate content if submitted
if st.session_state.blog_content is not None and st.session_state.submitted:
    system_prompt = """
    You are an experienced technical writer specializing in creating clear, 
    structured tutorials from existing technical content.
    """

    user_prompt = f"""
    Create a technical tutorial by filling out the article template {quickstart_template}
    by integrating content and code from the attached blog content {st.session_state.blog_content}. 
    
    In filling out the article template, please replace content specified by the brackets [].
    {f'Please use "{st.session_state.author_name}" as the author name.' if st.session_state.author_name else ''}
    {f'Please use "{extract_title(st.session_state.generated_blog)}" as the id.' if st.session_state.custom_title else ''}
    {'Please use the following categories: ' + ', '.join(st.session_state.selected_categories) if st.session_state.selected_categories else ''}
            
    Writing approach:
    - Professional yet accessible tone
    - Active voice
    - Direct reader address
    - Concise introduction focusing on value proposition

    Notes:
    - Please have the article title start with gerunds (Building, Performing, etc.)
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
    - Make sure that the generated output don't have enclosing ``` symbols at its top-most and bottom-post.
    - Please see if you can include links from the provided input blog that starts with https://docs.snowflake.com/en
      to the 'Articles:' segment of the Conclusion section.
    - If provided blog contains mention of Streamlit please add [Streamlit Documentation](https://docs.streamlit.io/)
      to the 'Documention' segment of the Conclusion section.
    - Add [Snowflake Documentation](https://docs.snowflake.com/) to the 'Documention' segment of the Conclusion section.
            
    Deliver the final output directly without meta-commentary.
    """

    st.subheader("Generated Tutorial")
    
    # Create a progress bar placeholder
    progress_bar = st.empty()
    progress_bar.progress(0, text="Starting tutorial generation...")
    
    try:
        for percent in range(0, 90, 10):
            time.sleep(0.1)
            progress_bar.progress(percent, text=f"Generating tutorial content... {percent}%")

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
        
        # Store the generated tutorial
        st.session_state.generated_blog = tutorial_content
        
        progress_bar.progress(100, text="Tutorial generation complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        # Display the tutorial content
        with st.expander('See generated tutorial'):
            st.code(st.session_state.generated_blog, language='markdown')

        # Get the title for the download filename
        title = extract_title(st.session_state.generated_blog)
        
        # Download button for zip file
        st.download_button(
            label="📥 Download ZIP",
            data=st.session_state.zip_data if st.session_state.zip_data else create_zip(),
            file_name=f"{title}.zip",
            mime="application/zip",
            key='download_button',
            help="Download the Quick Start tutorial with assets folder",
            on_click=handle_download
        )
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error generating tutorial: {str(e)}")
