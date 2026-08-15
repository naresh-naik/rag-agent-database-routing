"""Integration tests for rag_agent.pipeline orchestration."""

from unittest.mock import MagicMock, patch

from rag_agent.pipeline import PipelineResult, run_pipeline
from rag_agent.retriever import RetrievedDoc
from rag_agent.router import RoutingDecision


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
def test_run_pipeline_rag_path(mock_retrieve, mock_route):
    mock_route.return_value = RoutingDecision(
        database="products", reasoning="Query asks about product features"
    )
    mock_retrieve.return_value = [
        RetrievedDoc(text="TechPro X1 cost $999.", score=0.9, source="products")
    ]

    mock_groq = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "The TechPro X1 laptop costs $999."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_response

    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("How much is TechPro X1?", mock_qdrant, mock_embeddings, mock_groq)

    assert isinstance(result, PipelineResult)
    assert result.used_fallback is False
    assert result.answer == "The TechPro X1 laptop costs $999."
    assert len(result.docs) == 1
    assert result.routing.database == "products"


@patch("rag_agent.pipeline.route_query")
@patch("rag_agent.pipeline.retrieve")
@patch("rag_agent.pipeline.run_fallback")
def test_run_pipeline_fallback_path(mock_fallback, mock_retrieve, mock_route):
    mock_route.return_value = RoutingDecision(
        database="support", reasoning="Query asks about password reset"
    )
    mock_retrieve.return_value = []  # No docs found above score threshold
    mock_fallback.return_value = "Follow these web search instructions to reset password."

    mock_groq = MagicMock()
    mock_qdrant = MagicMock()
    mock_embeddings = MagicMock()

    result = run_pipeline("Password reset steps", mock_qdrant, mock_embeddings, mock_groq)

    assert isinstance(result, PipelineResult)
    assert result.used_fallback is True
    assert result.answer == "Follow these web search instructions to reset password."
    assert result.docs == []
    mock_fallback.assert_called_once_with(mock_groq, "Password reset steps")
