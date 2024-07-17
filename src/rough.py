import os
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import prettify_exec_info
openai_key = os.getenv("OPENAI_APIKEY")
load_dotenv()
graph_config = {
   "llm": {
      "api_key": openai_key,
      "model": "gpt-3.5-turbo",
   },
}
smart_scraper_graph = SmartScraperGraph(
   prompt="what is this about",
   # also accepts a string with the already downloaded HTML code
   source="https://www.geeksforgeeks.org/python-lists/",
   config=graph_config
)

result = smart_scraper_graph.run()
print(result)