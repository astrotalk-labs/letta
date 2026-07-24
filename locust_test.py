import random
import string

from locust import HttpUser, between, task

from letta.constants import BASE_TOOLS, DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.schemas.agent import AgentState, CreateAgent
from letta.schemas.letta_request import LettaRequest
from letta.schemas.letta_response import LettaResponse
from letta.schemas.memory import ChatMemory
from letta.schemas.message import MessageCreate, MessageRole
from letta.utils import get_human_text, get_persona_text


class LettaUser(HttpUser):
    wait_time = between(1, 5)
    token = None
    agent_id = None

    def on_start(self):
        # Create a user and get the token
        self.client.headers = {"Authorization": "Bearer password"}
        user_data = {"name": f"User-{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"}
        response = self.client.post("/v1/admin/users", json=user_data)
        response_json = response.json()
        print(response_json)
        self.user_id = response_json["id"]

        # create a token
        response = self.client.post("/v1/admin/users/keys", json={"user_id": self.user_id})
        self.token = response.json()["key"]

        # reset to use user token as headers
        self.client.headers = {"Authorization": f"Bearer {self.token}"}

        # @task(1)
        # def create_agent(self):
        # generate random name
        name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        request = CreateAgent(
            name=f"Agent-{name}",
            tools=BASE_TOOLS,
            memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text(DEFAULT_PERSONA)),
        )

        # create an agent
        with self.client.post("/v1/agents", json=request.model_dump(), headers=self.client.headers, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed to create agent: {response.text}")

            response_json = response.json()
            agent_state = AgentState(**response_json)
            self.agent_id = agent_state.id
            print("Created agent", self.agent_id, agent_state.name)

    @task(1)
    def send_message(self):
        messages = [MessageCreate(role=MessageRole("user"), content="hello")]
        request = LettaRequest(messages=messages)

        with self.client.post(
            f"/v1/agents/{self.agent_id}/messages", json=request.model_dump(), headers=self.client.headers, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to send message {response.status_code}: {response.text}")

            response = LettaResponse(**response.json())
            print("Response", response.usage)
