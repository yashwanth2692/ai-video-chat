import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faster_whisper
from fastapi.testclient import TestClient

model_mock = Mock()
with patch.object(faster_whisper, "WhisperModel", return_value=model_mock):
    import main


class MainTests(unittest.TestCase):
    def setUp(self):
        self.original_whisper = main.whisper
        self.original_transcribe = main.transcribe
        self.original_llm_stream = main.llm_stream
        self.original_chat = main.ollama.chat
        main.sessions.clear()
        main.whisper = Mock()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.whisper = self.original_whisper
        main.transcribe = self.original_transcribe
        main.llm_stream = self.original_llm_stream
        main.ollama.chat = self.original_chat

    def events(self, stream):
        return [json.loads(item) for item in stream]

    def test_clean_removes_markdown_characters(self):
        self.assertEqual(main.clean("  **Hello** _there_!  "), "Hello there!")

    def test_transcribe_joins_trimmed_segment_text(self):
        main.whisper.transcribe.return_value = (
            [SimpleNamespace(text=" Hello "), SimpleNamespace(text=" world. ")],
            None,
        )

        result = main.transcribe(b"audio")

        self.assertEqual(result, "Hello world.")
        audio = main.whisper.transcribe.call_args.args[0]
        self.assertIsInstance(audio, io.BytesIO)
        self.assertEqual(audio.read(), b"audio")
        self.assertEqual(
            main.whisper.transcribe.call_args.kwargs, {"vad_filter": True}
        )

    def test_llm_stream_yields_content_from_chunks(self):
        main.ollama.chat = Mock(
            return_value=[
                {"message": {"content": "Hello"}},
                {"message": {"content": " there."}},
            ]
        )

        self.assertEqual(list(main.llm_stream([{"role": "user"}], None)), ["Hello", " there."])
        self.assertEqual(main.ollama.chat.call_args.kwargs["stream"], True)

    def test_llm_stream_retries_without_image_after_response_error(self):
        error = main.ollama.ResponseError("Image input is unsupported", status_code=400)
        main.ollama.chat = Mock(
            side_effect=[
                error,
                [{"message": {"content": "Text reply."}}],
            ]
        )
        messages = [{"role": "user", "images": [b"frame"]}]
        fallback = [{"role": "user", "content": "Question"}]

        self.assertEqual(list(main.llm_stream(messages, fallback)), ["Text reply."])
        self.assertEqual(main.ollama.chat.call_count, 2)
        self.assertEqual(main.ollama.chat.call_args.kwargs["messages"], fallback)

    def test_auto_vision_includes_frame_for_visual_question(self):
        main.transcribe = Mock(return_value="What color is my shirt?")
        main.llm_stream = Mock(return_value=iter(["It is blue."]))

        events = self.events(main.turn_stream("session", b"audio", b"jpeg", "auto"))

        self.assertEqual(events[1], {"type": "vision", "used": True})
        messages = main.llm_stream.call_args.args[0]
        self.assertEqual(messages[-1]["images"], [b"jpeg"])

    def test_vision_off_does_not_send_frame(self):
        main.transcribe = Mock(return_value="What color is my shirt?")
        main.llm_stream = Mock(return_value=iter(["It is blue."]))

        events = self.events(main.turn_stream("session", b"audio", b"jpeg", "off"))

        self.assertEqual(events[1], {"type": "vision", "used": False})
        messages = main.llm_stream.call_args.args[0]
        self.assertNotIn("images", messages[-1])

    def test_empty_transcription_skips_model_call(self):
        main.transcribe = Mock(return_value="")
        main.llm_stream = Mock()

        events = self.events(main.turn_stream("session", b"audio", None, "auto"))

        self.assertEqual(events, [{"type": "empty"}])
        main.llm_stream.assert_not_called()

    def test_transcription_exception_is_returned_as_error_event(self):
        main.transcribe = Mock(side_effect=RuntimeError("invalid audio"))

        events = self.events(main.turn_stream("session", b"audio", None, "auto"))

        self.assertEqual(
            events,
            [{"type": "error", "text": "Transcription failed: invalid audio"}],
        )

    def test_turn_stream_emits_sentences_and_saves_history(self):
        main.transcribe = Mock(return_value="How are you?")
        main.llm_stream = Mock(return_value=iter(["I am doing well.", " How are you?"]))

        events = self.events(main.turn_stream("session", b"audio", None, "auto"))

        self.assertEqual(
            events,
            [
                {"type": "transcript", "text": "How are you?"},
                {"type": "vision", "used": False},
                {"type": "sentence", "text": "I am doing well."},
                {"type": "sentence", "text": "How are you?"},
            ],
        )
        self.assertEqual(
            main.sessions["session"],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am doing well. How are you?"},
            ],
        )

    def test_reset_endpoint_clears_session(self):
        main.sessions["session"] = [{"role": "user", "content": "Hello"}]

        response = self.client.post("/api/reset", data={"session_id": "session"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertNotIn("session", main.sessions)


if __name__ == "__main__":
    unittest.main()
