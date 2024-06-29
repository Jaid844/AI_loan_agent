from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
db="hellw"
llm="fffe"

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
print(tools)