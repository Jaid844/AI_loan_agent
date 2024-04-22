from graph import WorkFlow
from pprint import pprint
app = WorkFlow().app


# Run
inputs = {

        "name":"James",
       "session_id":"cutomer_james",


}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")


