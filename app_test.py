import unittest
from app import CompletionRequest, ChatCompletionItem
from pydantic import ValidationError

class TestDeserialization(unittest.TestCase):
    def test_serialize_request(self):
        request_string: str = '''
        {
            "data": [
                [0, "systemPrompt", "userPrompt", {"inputValues": "someValue"}, {"metadata": "someMetadata"}]
            ]
        } 
        '''

        try:
            request: CompletionRequest = CompletionRequest.model_validate_json(request_string)

        except ValidationError as e:
            self.fail(f"Serialization failed with error: {e}")

        self.assertEqual(request.systemPrompt, "systemPrompt")
        self.assertEqual(request.userPrompt, "userPrompt")
        self.assertEqual(request.inputValues, {"inputValues": "someValue"})
        self.assertEqual(request.metadata, {"metadata": "someMetadata"})


if __name__ == '__main__':
    unittest.main()