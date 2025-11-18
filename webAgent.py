from openai import OpenAI
import time

class WebAgent:
    """
    Reusable Web Browsing Agent using OpenAI Assistants + Browser Tool.
    """
    
    def __init__(self, assistant_id=None, model="gpt-5.1"):
        self.client = OpenAI()
        self.model = model
        
        if assistant_id:
            self.assistant_id = assistant_id
        else:
            # Create new assistant with browser tool
            assistant = self.client.beta.assistants.create(
                name="ReusableWebAgent",
                model=self.model,
                instructions=(
                    "You are a web-browsing agent. You can open URLs, click elements, "
                    "fill forms, extract structured data, and navigate webpages."
                ),
                tools=[{"type": "browser"}]
            )
            self.assistant_id = assistant.id
        
        # Create an interaction thread
        self.thread = self.client.beta.threads.create()

    # -------------------------------------------------------------------------
    # Internal utility to send message + poll run
    # -------------------------------------------------------------------------
    def _send_and_run(self, user_message):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_message
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id
        )
        return self._latest_response()

    # -------------------------------------------------------------------------
    # Retrieve last agent message
    # -------------------------------------------------------------------------
    def _latest_response(self):
        msgs = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        for msg in msgs.data:
            if msg.role == "assistant":
                return msg.content[0].text.value
        return None

    # -------------------------------------------------------------------------
    # High-level agent actions
    # -------------------------------------------------------------------------
    def go(self, url):
        """Open a webpage."""
        return self._send_and_run(f"Open {url}")

    def click(self, selector=None, text=None, index=0):
        """
        Click element by CSS selector or text.
        Example:
            agent.click(selector="button.submit")
            agent.click(text="Login")
        """
        if selector:
            msg = f"Click the element with CSS selector '{selector}'."
        else:
            msg = f"Click the element that has visible text '{text}'."
        return self._send_and_run(msg)

    def type(self, selector, text):
        """Type in input box."""
        return self._send_and_run(
            f"Type '{text}' into the input with CSS selector '{selector}'."
        )

    def extract(self, description):
        """
        Ask agent to extract data in structured JSON.
        Example:
            agent.extract('Extract all product names & prices as JSON list.')
        """
        return self._send_and_run(description)

    def task(self, instruction):
        """
        Free-form multi-step task.
        """
        return self._send_and_run(instruction)

    # -------------------------------------------------------------------------
    # Convenience shortcut (print response)
    # -------------------------------------------------------------------------
    def ask(self, message):
        """Send a message and print the agent's response."""
        resp = self._send_and_run(message)
        print(resp)
        return resp

