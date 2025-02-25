from versionhq.agent.model import Agent
from versionhq.llm.model import DEFAULT_MODEL_NAME

"""
In-house agents to be called across the project.
[Rules] In house agents have names and roles that start with `vhq_`. No customization allowed by client.
"""

vhq_client_manager = Agent(
    role="vhq-Client Manager",
    goal="Efficiently communicate with the client on the task progress",
    llm=DEFAULT_MODEL_NAME,
    maxit=1,
    max_retry_limit=1,
    with_memory=True,
)


vhq_task_evaluator = Agent(
    role="vhq-Task Evaluator",
    goal="score the output according to the given evaluation criteria, taking a step by step approach.",
    llm=DEFAULT_MODEL_NAME,
    llm_config=dict(top_p=0.8, top_k=30, max_tokens=5000, temperature=0.9),
    maxit=1,
    max_retry_limit=1,
    with_memory=True # refer past eval records of similar tasks
)


vhq_formation_planner = Agent(
    role="vhq-Formation Planner",
    goal="Plan a formation of agents based on the given task descirption.",
    llm="gemini/gemini-2.0-flash-exp",
    llm_config=dict(top_p=0.8, topK=40, temperature=0.9),
    maxit=1,
    max_retry_limit=1,
    knowledge_sources=[
        "Solo is a formation where a single agent with tools, knowledge, and memory handles tasks indivudually. When self-learning mode is on - it will turn into Random formation. Typical usecase is an email agent drafts promo message for the given audience using their own knowledge.",
        "Supervising is a formation where the leader agent gives directions, while sharing its knowledge and memory with subbordinates.Subordinates can be solo agents or networks. Typical usecase is that the leader agent strategizes an outbound campaign plan and assigns components such as media mix or message creation to subordinate agents.",
        "Network is a formation where multple agents can share tasks, knowledge, and memory among network members without hierarchy. Typical usecase is that an email agent and social media agent share the product knowledge and deploy multi-channel outbound campaign. ",
        "Random is a formation where a single agent handles tasks, asking help from other agents without sharing its memory or knowledge. Typical usecase is that an email agent drafts promo message for the given audience, asking insights on tones from other email agents which oversee other customer clusters, or an agent calls the external, third party agent to deploy the campaign.",
    ]
)


vhq_agent_creator = Agent(
    role="vhq-Agent Creator",
    goal="build an agent that can handle the given task",
    llm="gemini/gemini-2.0-flash-exp",
    maxit=1,
    max_retry_limit=1,
)
