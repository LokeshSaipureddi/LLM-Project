from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RAGQueryRequest(BaseModel):
    """
    Request model for RAG query
    """
    query: str = Field(..., description="User query for document search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results")

class RAGQueryResponse(BaseModel):
    """
    Response model for RAG query
    """
    results: List[Dict] = Field(default_factory=list, description="Retrieved document results")
    status: str = Field(..., description="Status of the query")
    message: Optional[str] = Field(default=None, description="Error message if any")

class AgentReportRequest(BaseModel):
    """
    Request model for agent report generation
    """
    company_name: str = Field(..., description="Name of the company for report generation")
    include_financials: bool = Field(default=True, description="Include financial details")
    include_charts: bool = Field(default=True, description="Include financial charts")

class AgentReportResponse(BaseModel):
    """
    Response model for agent report generation
    """
    report: Dict = Field(default_factory=dict, description="Generated financial report")
    status: str = Field(..., description="Status of report generation")
    message: Optional[str] = Field(default=None, description="Error message if any")