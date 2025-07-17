from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


history = FileChatMessageHistory("chat_history.json")

llm = ChatOpenAI(
    model_name = "gpt-4.1-nano",
     temperature = 1 )

memory = ConversationBufferMemory(

    memory_key = "chat_history",
    chat_memory=history,
    return_messages = True,
)

prompt = ChatPromptTemplate(
    input_variables = ['content'],
    messages = [

        SystemMessage( 
            content = "You are chatbot having conversation with a human. "),
        MessagesPlaceholder(variable_name = "chat_history"),
        HumanMessagePromptTemplate.from_template('{content}') 
    ]
)

chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
    verbose = False
)

while True:
    content = input("You: ")
    if content.lower() in ['exit', 'quit', 'bye']:
        print('Goodbye!')
        break

    response = chain.run({"content": content})
    print(response)



