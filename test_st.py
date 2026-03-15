import streamlit as st
import inspect
import json

doc = st.chat_input.__doc__
print("DOCSTART")
print(doc)
print("DOCEND")
