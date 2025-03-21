import streamlit as st

st.title("Test Streamlit App")
st.write("This is a simple test app to check if Streamlit is working correctly.")

st.sidebar.write("Sidebar test")

if st.button("Click me"):
    st.success("Button clicked!")
