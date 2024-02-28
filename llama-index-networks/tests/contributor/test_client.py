from unittest.mock import MagicMock, patch
from llama_index.networks import (
    ContributorClient,
    ContributorClientSettings,
)
from llama_index.core.settings import Settings


@patch("llama_index.networks.contributor.client.requests")
def test_contributor_client(mock_requests):
    # arrange
    result_mock = MagicMock()
    result_mock.status_code = 200
    result_mock.json.return_value = {
        "response": "Mock response",
        "souce_nodes": [],
        "metadata": None,
    }
    mock_requests.post.return_value = result_mock

    settings = ContributorClientSettings(api_url="fake-url")
    client = ContributorClient(
        config=settings, callback_manager=Settings.callback_manager
    )

    # act
    res = client.query("Does this work?")
    print(res)

    # assert
    mock_requests.post.assert_called_once()
    args, kwargs = mock_requests.post.call_args
    assert kwargs == {
        "json": {"query": "Does this work?", "api_key": None},
        "headers": {},
    }
    assert args[0] == "fake-url/api/query"
    assert res.response == "Mock response"
    assert res.metadata == {"score": None}
