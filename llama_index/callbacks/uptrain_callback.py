from typing import Any, Dict, List, Optional

from uptrain import APIClient, Evals, Settings

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType


class UpTrainDataSchema:
    """UpTrain data schema."""

    def __init__(
        self, question: str = "", context: str = "", response: str = ""
    ) -> None:
        """Initialize the UpTrain data schema."""
        self.question = question
        self.context = context
        self.response = response


class UpTrainCallbackHandler(BaseCallbackHandler):
    """UpTrain callback handler."""

    def __init__(self) -> None:
        """Initialize the UpTrain callback handler."""
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self.uptrain_data_schema = UpTrainDataSchema()

        settings = Settings(uptrain_access_token="up-***********************")
        self.uptrain_client = APIClient(settings=settings)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Any = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        if event_type is CBEventType.TEMPLATING:
            self.uptrain_data_schema.question = payload["template_vars"]["query_str"]
            self.uptrain_data_schema.context = payload["template_vars"]["context_str"]

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Any = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        # Response
        if event_type is CBEventType.SYNTHESIZE:
            self.uptrain_data_schema.response = payload["response"].response
            self.uptrain_client.log_and_evaluate(
                project_name="llama",
                data=[
                    {
                        "question": self.uptrain_data_schema.question,
                        "context": self.uptrain_data_schema.context,
                        "response": self.uptrain_data_schema.response,
                    }
                ],
                checks=[
                    Evals.CONTEXT_RELEVANCE,
                    Evals.FACTUAL_ACCURACY,
                    Evals.RESPONSE_COMPLETENESS,
                ],
            )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return super().start_trace(trace_id)

    def end_trace(
        self, trace_id: str | None = None, trace_map: Dict[str, List[str]] | None = None
    ) -> None:
        return super().end_trace(trace_id, trace_map)
