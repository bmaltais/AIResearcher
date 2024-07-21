# AIResearcher

This is a simple RAG/AI system designed for ease of use. It utilizes Chromadb and Gemini 1.5, both of which offer free usage at low levels. The system is intended to be straightforward yet sophisticated enough to import hundreds of books on a subject and generate quality AI-driven answers. Future iterations may explore other AI tools, and it would be relatively simple to adapt it for use with Ollama, Claude, GPT-4, or other AI models.

## Installation (Windows)

1. Clone the repository
2. Navigate to the cloned repository directory:
   
   `cd <cloned-repo-name>`
   
3. Create a new virtual environment:
   
   ```pwsh
   python -m venv venv
   ./venv/Scripts/activate
   ```
   
4. Install required packages:
   
   `pip install -r requirements.txt`
   
5. Obtain an API key from Google:
   [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

## Usage (Windows)

1. Collect source material
   - The script supports importing PDF, EPUB, TXT, and Markdown files.
   - Note: This is a text-based system and won't work with photos, graphs, etc.
   - Organize books by a single subject for optimal results.
   - Limit sources to quality materials you want to provide answers.
   - For divergent viewpoints, consider creating separate collections.

2. Import source material to Chroma:
   
   `python .\importall_embeds.py --collection books --document-directory "H:\open-webui\data\docs\books\"`
   

3. Query the system:
   
   `python .\research-main_embeds.py --collection books`
   
   - Type your questions into the terminal.
   - Read responses in the terminal or in Obsidian for a better experience.
   - The Admonition plugin for Obsidian can enhance formatting of the output.
   - Sources are currently set to output into the markdown.
