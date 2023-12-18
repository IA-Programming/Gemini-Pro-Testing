from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents import load_tools
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_log_to_str

from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, max_output_tokens=2048, metadata={'streaming': True})

# from langchain import hub
# obj = hub.pull("hwchase17/react-chat")

template = """Assistant is a large language model trained by Google.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

prompt = ChatPromptTemplate.from_template(template=template)

tools = load_tools(["ddg-search", "llm-math"], llm=gemini)

prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

llm_with_stop = gemini.bind(stop=["\nObservation"])
chatAgent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)
agent = AgentExecutor(
    agent=chatAgent,
    tools=tools,
    memory=ConversationBufferMemory(memory_key="chat_history", output_key="output"),
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    verbose=True,
)

response = agent.invoke({"input": "what News of Ai we have today"})

print(response)

# Resources:
# https://python.langchain.com/docs/modules/agents/agent_types/react
# https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent

# Prompts:
# [Adding Memory to a chat model-based](https://python.langchain.com/docs/modules/memory/adding_memory#adding-memory-to-a-chat-model-based-llmchain)
# [ChatPromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/#chatprompttemplate)
# [Attaching Function Call information](https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser#attaching-function-call-information)