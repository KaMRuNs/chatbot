# Save this as 'visualize_graph.py' and run it
from agents.action_agent import create_react_agent, TOOLS
from utils.llm import get_llm
import os

# Create the same agent structure
llm = get_llm()
agent = create_react_agent(model=llm, tools=TOOLS)

# Generate and save the image
try:
    # Requires: pip install pygraphviz OR graphviz
    graph_image = agent.get_graph().draw_mermaid_png()
    with open("langgraph_architecture.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved as 'langgraph_architecture.png'!")
except Exception as e:
    print(f"To generate the image, you might need to install 'grandalf': pip install grandalf")
    # Alternative: Print the Mermaid text which you can paste into https://mermaid.live
    print("\n--- Copy & Paste this into mermaid.live ---")
    print(agent.get_graph().draw_mermaid())
