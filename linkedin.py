import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain.chains import LLMChain

from langchain.tools import tool
from langchain import hub
from langchain_core.messages import HumanMessage,ToolMessage,SystemMessage
# from langgraph.prebuilt import create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel

load_dotenv()
os.environ['GOOGLE_API_KEY'] = ""
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",

)




prompt=PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: 

---

### Summary of JSON Data:

**Data Overview:**
- **Total Entries:** [number_of_entries]
- **Key Highlights:**
  - **[key_1]:** [value_1]
  - **[key_2]:** [value_2]
  - **[key_3]:** [value_3]
  - *(Continue as needed based on JSON structure)*

**Detailed Breakdown:**
- **[first_entry_key]:** 
  - **Sub-key 1:** [sub_value_1]
  - **Sub-key 2:** [sub_value_2]
  - **Sub-key 3:**[sub_value_3]
  - *(Continue as needed)*

- **[second_entry_key]:** 
  - **Sub-key 1:** [sub_value_1]
  - **Sub-key 2:** [sub_value_2]
  - **Sub-key 3:** [sub_value_3]
  - *(Continue as needed)*

*Add more sections depending on the complexity and size of the JSON data.*

---

Example Final Answer:
---

### Summary of JSON Data:

**Data Overview:**
- **Total Entries:** 5
- **Key Highlights:**
  - **Name:** "LangChain"
  - **Year:** 2023
  - **Rating:** 9.5

**Detailed Breakdown:**
- **Entry 1:** 
  - **Title:** "Introduction to LangChain"
  - **Author:** "John Doe"
  - **Summary:** "An in-depth guide to LangChain usage."

- **Entry 2:** 
  - **Title:** "Advanced LangChain Techniques"
  - **Author:** "Jane Smith"
  - **Summary:** "Exploring the advanced features of LangChain."

---

This summary provides a structured and clear representation of the JSON data received from the tools.

Question: {input}
Thought:{agent_scratchpad}
""")

    
# prompt = PromptTemplate.from_template(template)
# from langchain import hub
# prompt = hub.pull("hwchase17/react")

mock_url="https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/78233eb934aa9850b689471a604465b188e761a0/eden-marco.json"
@tool
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """Scrape information from LinkedIn profiles.

    Args:
        linkedin_profile_url (str): A LinkedIn profile URL.
        mock (bool): Flag to use mock data.

    Returns:
        dict: A dictionary with the scraped LinkedIn profile data.
    """
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/78233eb934aa9850b689471a604465b188e761a0/eden-marco.json"
        response = requests.get(linkedin_profile_url, timeout=10)
        data = response.json()
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
        response = requests.get(
            api_endpoint,
            params={"url": linkedin_profile_url},
            headers=header_dic,
            timeout=10,
        )
        data = response.json()

    # data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data



tools = [scrape_linkedin_profile]
agent= create_react_agent(llm, tools, prompt=prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)


result=agent_executor.invoke({"input": mock_url})
print(result['output'])


# if __name__ == "__main__":
#     print(
#         scrape_linkedin_profile(
#             linkedin_profile_url="https://www.linkedin.com/in/abhay-tyagi-561b22207/",
#         )
#     )