"""
Main FastAPI application for the Medical AI Assistant.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

from app.api.routes import router
from config.settings import settings

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered medical document Q&A assistant using RAG and Google Gemini",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Medical AI Assistant"])

# Mount static files for UI (if needed)
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.APP_NAME}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .endpoint {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
            .method {{ font-weight: bold; color: #e74c3c; }}
            .url {{ font-family: monospace; background-color: #34495e; color: white; padding: 2px 6px; border-radius: 3px; }}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{settings.APP_NAME}</h1>
            <p><strong>Version:</strong> {settings.APP_VERSION}</p>
            <p>AI-powered medical document Q&A assistant using RAG (Retrieval-Augmented Generation) and Google Gemini.</p>
            
            <h2>Features</h2>
            <ul>
                <li>Upload PDF medical documents</li>
                <li>Ask questions about uploaded documents</li>
                <li>Get AI-generated answers with source citations</li>
                <li>Manage document library</li>
                <li>Real-time document processing</li>
            </ul>
            
            <h2>API Documentation</h2>
            <p>
                <a href="/docs" target="_blank">Interactive API Documentation (Swagger UI)</a><br>
                <a href="/redoc" target="_blank">Alternative API Documentation (ReDoc)</a>
            </p>
            
            <h2>Quick Start</h2>
            <div class="endpoint">
                <strong>1. Upload a PDF document:</strong><br>
                <span class="method">POST</span> <span class="url">/api/v1/upload</span><br>
                Upload a PDF file to process and index for Q&A.
            </div>
            
            <div class="endpoint">
                <strong>2. Ask a question:</strong><br>
                <span class="method">POST</span> <span class="url">/api/v1/ask</span><br>
                Ask questions about your uploaded documents.
            </div>
            
            <div class="endpoint">
                <strong>3. View documents:</strong><br>
                <span class="method">GET</span> <span class="url">/api/v1/documents</span><br>
                List all uploaded documents and their status.
            </div>
            
            <div class="endpoint">
                <strong>4. Check system health:</strong><br>
                <span class="method">GET</span> <span class="url">/api/v1/health</span><br>
                Verify that all services are running properly.
            </div>
            
            <h2>Requirements</h2>
            <ul>
                <li>Google Gemini API Key (set GEMINI_API_KEY environment variable)</li>
                <li>PDF documents for upload and Q&A</li>
            </ul>
            
            <p><em>Built with FastAPI, Google Gemini, FAISS, and Sentence Transformers.</em></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/v1")
async def api_info():
    """API information endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Medical AI Assistant API",
        "endpoints": {
            "upload": "/api/v1/upload",
            "ask": "/api/v1/ask",
            "documents": "/api/v1/documents",
            "health": "/api/v1/health",
            "stats": "/api/v1/stats"
        }
    }

if __name__ == "__main__":
    # Check if GEMINI_API_KEY is set
    if not settings.GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please set your Google Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        print()
    
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Server will be available at: http://{settings.HOST}:{settings.PORT}")
    print(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    print()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 