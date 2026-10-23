# RAG Document Q&A Assistant

A Python command-line Retrieval-Augmented Generation (RAG) app that answers questions using local PDF documents and shows source page citations.

## Requirements
- Python 3.11 or 3.12
- OpenAI API key with API access/billing configured
- Internet access for package/model downloads and API calls

## Setup (Windows / VS Code)
1. Extract this project ZIP.
2. Open the extracted `rag_document_assistant` folder in VS Code.
3. Add text-based PDF files to the `docs` folder.
4. Rename `.env.example` to `.env`, then add your real OpenAI API key.
5. Open Terminal > New Terminal and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use Command Prompt and run:
```cmd
.venv\Scripts\activate.bat
```

Ask questions in the terminal. Type `exit` to stop.

## Notes
- The embedding model runs locally after its first download.
- OpenAI API calls may incur charges.
- Scanned/image-only PDFs need OCR before this program can read them.
- The local ChromaDB collection is rebuilt each time the app starts.
- Answers are constrained by instructions to use retrieved context only, but citations and answers should still be checked.
- Never share or commit your `.env` file.
