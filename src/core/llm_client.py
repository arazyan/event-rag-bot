import sys
import os
import ollama
import json
from pydantic import ValidationError

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.prompt import SYSTEM_PROMPT
from src.utils.schema import EventModel


class EventExtractor:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model = model_name

    def process_post(self, text: str, event_id: str = "") -> EventModel | None:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Текст поста:\n{text}"},
                ],
                format="json",
            )

            event_data = json.loads(response["message"]["content"])
            event_data["event_id"] = event_id  # TODO: id will be extracted via telegram

            return EventModel(**event_data)

        except ValidationError as e:
            # TODO: add logging
            print(f"Error validating post: {event_id}\n{e}")
            return None
        except Exception as e:
            # TODO: add logging
            print(f"An unexpected error occurred: {e}")
            return None


if __name__ == "__main__":
    EXAMPLE_POST = """\
    ADD POST HERE
    """
    if "ADD POST HERE" in EXAMPLE_POST:
        print("Update EXAMPLE_POST variable in the src/llm_client.py")
    extractor = EventExtractor(model_name="qwen2.5:1.5b")  # NOTE: this is tiny model
    event = extractor.process_post(text=EXAMPLE_POST, event_id="1234567890")
    if event:
        print(event.model_dump_json(indent=2))
