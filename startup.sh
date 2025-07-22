#!/bin/bash
echo "Starting server on port $PORT"
uvicorn rag_chatbot:app --host 0.0.0.0 --port $PORT