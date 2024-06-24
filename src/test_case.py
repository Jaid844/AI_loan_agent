from main import *
import pytest
from pytest import MonkeyPatch
import unittest
import unittest.mock
from tools import monthly_payment_1, loan_embedding_model, monthly_payment
from audio import audio_node
from unittest.mock import patch, Mock

c = audio_node()


@pytest.fixture
def name_and_rate() -> dict:
    """Create a cache backed embeddings."""
    data = {"name": "James", "rate": 10}
    return data


def mock_tool_response(name, rate):
    return "This will be the last 299.997  payment for the customer James"


def test_process_data(monkeypatch, name_and_rate):
    name = name_and_rate['name']
    rate = name_and_rate['rate']
    monkeypatch.setattr('tools.monthly_payment', mock_tool_response)
    result = monthly_payment_1(name, rate)
    assert result == "This will be the last 299.997  payment for the customer James"


def audio(text):
    response = c.streamed_audio(text)
    return response.json()


class TestData(unittest.TestCase):
    @patch('request.get')
    def test_get_user_data(self, moc_get):
        mock_response = Mock()
        response_dict = {"audio_repsonse": "Hellow"}
        mock_response.json.return_values = response_dict
        moc_get.return_value=mock_response
        moc_get.assert_called_with()
