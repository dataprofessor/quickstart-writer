import streamlit as st
import requests
import os
import json
import anthropic
import zipfile
import io
import time
from openai import OpenAI

# Initialize session state variables
if 'blog_content' not in st.session_state:
    st.session_state.blog_content = None
if 'generated_blog' not in st.session_state:
    st.session_state.generated_blog = None
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None

# Verify API keys
if 'OPENAI_API_KEY' not in st.secrets:
    st.error("Please set your OpenAI API key in Streamlit secrets as OPENAI_API_KEY")
    st.stop()
if 'ANTHROPIC_API_KEY' not in st.secrets:
    st.error("Please set your Anthropic API key in Streamlit secrets as ANTHROPIC_API_KEY")
    st.stop()

# Template for the quickstart guide
quickstart_template = '''
id: [unique_identifier_with_underscores]
summary: [One to two sentences describing what this guide covers]
categories: [comma-separated list: e.g., featured,getting-started,data-engineering]
environments: web
status: Published
feedback link: https://github.com/Snowflake-Labs/sfguides/issues
tags: [Comma-separated list of relevant technologies and concepts]
authors: [Your Name]


# [Article Title]

## Overview
Duration: [X]

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
- [Resource link 1 with description]
- [Resource link 2 with description]
- [Resource link 3 with description]

### Documentation
- [Relevant documentation link 1]
- [Relevant documentation link 2]

### Additional Reading
- [Blog/article link 1]
- [Blog/article link 2]
'''

def create_zip():
    """Creates markdown file and returns zipped bytes."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if 'generated_blog' in st.session_state:
            zip_file.writestr('tutorial.md', st.session_state.generated_blog)
    
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

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Quick Start",
    page_icon="⏩",
    layout="wide"
)

with st.sidebar:
    st.title("⏩ Write Quick Start")
    st.warning(
        "Transform your technical blog posts into Quick Start tutorials."
    )
    
    # File uploader for blog content
    uploaded_file = st.file_uploader(
        "Upload your blog content (markdown or text file)",
        type=['md', 'txt'],
        help="Upload a file containing your blog content"
    )
    
    if uploaded_file is not None:
        st.session_state.blog_content = uploaded_file.getvalue().decode('utf-8')
    
    st.subheader("⚙️ Settings")
    llm_model = st.selectbox(
        "Select a model",
        ("o1-mini", "gpt-4-turbo", "claude-3-5-sonnet-20241022")
    )

    if st.session_state.blog_content:
        st.button(
            "Reset All",
            type="primary",
            on_click=reset_callback,
            use_container_width=True
        )

if not uploaded_file:
    st.info("Please upload your blog content file in the sidebar!", icon="👈")

if st.session_state.blog_content is not None:
    system_prompt = """
    You are an experienced technical writer specializing in creating clear, 
    structured tutorials from existing technical content.
    """

    user_prompt = f"""
    Create a technical tutorial by filling out the article template ({quickstart_template})
    by integrating content and code from the attached blog content ({st.session_state.blog_content}). 
    
    In filling out the article template, please replace content specified by the brackets [].
            
    Writing approach:
    - Professional yet accessible tone
    - Active voice
    - Direct reader address
    - Concise introduction focusing on value proposition

    Notes:
    - In the Resources section, if you don't have the URL, please just replace it with '#-REPLACE-WITH-URL'
    - For the Duration, please give an estimate for reading and completing the task mentioned in each section.
            
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
            st.markdown(st.session_state.generated_blog)
                        
        with st.expander("Generated Tutorial (Markdown)"):
            st.code(st.session_state.generated_blog, language='markdown')

        # Download button for zip file
        st.download_button(
            label="📥 Download ZIP file",
            data=st.session_state.zip_data if st.session_state.zip_data else create_zip(),
            file_name="tutorial.zip",
            mime="application/zip",
            key='download_button',
            help="Download the tutorial as markdown file in a zip",
            on_click=handle_download
        )
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error generating tutorial: {str(e)}")
