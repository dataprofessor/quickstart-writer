[Previous imports and constants remain the same until the UI section...]

# Set up the Streamlit page
st.set_page_config(
    page_title="Write Quickstarts",
    page_icon="⏩",
    layout="wide"
)

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
            placeholder="https://github.com/username/repo/blob/main/file.ipynb or repository URL"
        )
        
        st.caption("""
        You can provide either:
        - Direct link to a .ipynb file
        - Repository URL (will automatically find .ipynb files)
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
