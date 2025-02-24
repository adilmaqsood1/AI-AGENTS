from fastapi import FastAPI, HTTPException
from schemas.models import CrawlRequest, CrawlResponse
from services.keyword_analysis_workflow import KeywordAnalysisWorkflow
from dotenv import load_dotenv
import os 

app = FastAPI()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the keyword analysis API!"}

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    workflow = KeywordAnalysisWorkflow(GROQ_API_KEY)

    try:
        result = workflow.run(request.url)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))