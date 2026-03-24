import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Photo Editor", layout="wide")

# CUSTOM UI STYLE
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1 {
    text-align: center;
    font-size: 3rem;
}
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}
.stDownloadButton button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<h1>📸 Photo Editor</h1>
<p style='text-align:center;'>Enhance your images with real-time filters</p>
""", unsafe_allow_html=True)

# FUNCTIONS

def load_image(file):
    return np.array(Image.open(file).convert("RGB"))

def resize_image(img, width, height):
    h, w = img.shape[:2]
    if width < w or height < h:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=inter)

def adjust_bc(img, b, c):
    img = img.astype(np.float32)
    img = img * c + b
    return np.clip(img, 0, 255).astype(np.uint8)

def blur(img, k):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def warm(img):
    img = img.astype(np.float32)
    img[:,:,2] += 20
    img[:,:,0] -= 10
    return np.clip(img,0,255).astype(np.uint8)

def portrait(img):
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = (X - w//2)**2 + (Y - h//2)**2 <= (min(h,w)//3)**2
    mask = cv2.GaussianBlur(mask.astype(np.float32),(51,51),0)
    blur_img = cv2.GaussianBlur(img,(31,31),0)
    mask = np.stack([mask]*3,-1)
    return (img*mask + blur_img*(1-mask)).astype(np.uint8)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def encode(img):
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

# UPLOAD
file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if file:
    img = load_image(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # SIDEBAR CONTROLS
    st.sidebar.title("🎛️ Controls")

    with st.sidebar.expander("📐 Resize", True):
        keep = st.checkbox("Keep Ratio", True)
        new_w = st.slider("Width", 50, 2000, w)
        if keep:
            new_h = int(new_w * h / w)
        else:
            new_h = st.slider("Height", 50, 2000, h)

    with st.sidebar.expander("⚙️ Adjust"):
        bright = st.slider("Brightness", -100, 100, 0)
        cont = st.slider("Contrast", 0.5, 3.0, 1.0)

    with st.sidebar.expander("🎨 Filters"):
        blur_on = st.checkbox("Blur")
        blur_k = st.slider("Blur Strength", 1, 25, 5)
        sharp_on = st.checkbox("Sharpen")
        warm_on = st.checkbox("Warm")
        gray_on = st.checkbox("Grayscale")
        port_on = st.checkbox("Portrait Blur")

    # PROCESS
    out = img.copy()

    out = resize_image(out, new_w, new_h)
    out = adjust_bc(out, bright, cont)

    if not gray_on and warm_on:
        out = warm(out)

    if blur_on:
        out = blur(out, blur_k)

    if sharp_on:
        out = sharpen(out)

    if port_on:
        out = portrait(out)

    if gray_on:
        out = gray(out)

    # DISPLAY
    st.markdown("## 🖼️ Editor Workspace")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)

    with col2:
        st.markdown("### ✨ Edited")
        if len(out.shape) == 2:
            st.image(out, width=400, channels="GRAY")
        else:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width=400)

    st.markdown("---")

    # DOWNLOAD
    st.download_button(
        "⬇️ Download Image",
        data=encode(out),
        file_name="edited.png",
        mime="image/png"
    )

    st.success("✅ Image processed successfully!")