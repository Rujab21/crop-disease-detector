import streamlit as st

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱", layout="wide")

st.title("🌱 Crop Disease Detector")
st.markdown("""
Welcome to the **AI-powered Crop Disease Detector**!  
This tool helps identify crop diseases using deep learning models (ConvNeXt-Tiny & EfficientNet).  

👉 Use the sidebar to explore:
- **About the app**  
- **Model details**  
- **Test the model with your images**
""")


col1, col2, col3 = st.columns(3)

with col1:
    st.image("imagesforfrontpage/crop1.jpeg",
             caption="Make Crop farming Sustainable", use_container_width=True)

with col2:
    st.image("imagesforfrontpage/crop2.jpeg",
             caption="Detect Diseases faster", use_container_width=True)

with col3:
    st.image("imagesforfrontpage/crop3.jpeg",
             caption="Reliable", use_container_width=True)

st.markdown("---")
st.info("💡 Try uploading your own crop images on the **Test Model** page!")

