import logging
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
    def __init__(self, model_name="qwen2.5:1.5b"):
        self.model = model_name
        self.async_client = ollama.AsyncClient()

    async def process_post(self, text: str, event_id: str = "") -> EventModel | None:
        try:
            response = await self.async_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Текст поста:\n{text}"},
                ],
                format="json",
            )

            content = response["message"]["content"]
            event_data = json.loads(content)
            event_data["event_id"] = event_id

            return EventModel(**event_data)

        except ValidationError as e:
            logging.error(f"Error validating post: {event_id}\n{e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None


if __name__ == "__main__":
    EXAMPLE_POST = """\
    ADD POST HERE
    """
    if "ADD POST HERE" in EXAMPLE_POST:
        logging.info("[INFO] Update EXAMPLE_POST variable in the src/llm_client.py")
    else:
        import asyncio
        async def test():
            extractor = EventExtractor()
            event = await extractor.process_post(text=EXAMPLE_POST, event_id="1")
            if event:
                print(event.model_dump_json(indent=2))
        asyncio.run(test())
