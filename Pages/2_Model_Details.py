import streamlit as st

st.title("🧠 Model Details")

st.write("""
We use two different CNN-based models:  

1. **EfficientNetB3**  
   - Balanced efficiency and accuracy.  
   - Pretrained on ImageNet, fine-tuned on crop disease dataset.  

2. **ConvNeXt-Tiny**  
   - A modern architecture inspired by transformers.  
   - Handles complex crop images with high accuracy.  

Both models support **Grad-CAM visualization** to highlight important regions in images.
""")
