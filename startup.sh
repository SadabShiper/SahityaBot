#!/bin/bash
uvicorn rag_chatbot:app --host 0.0.0.0 --port ${PORT:-8000}