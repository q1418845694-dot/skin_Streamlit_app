import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm


class Config:
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    IMG_SIZE = 224

    MODEL_WEIGHTS = "best_model.pth"
    CSV_PATH = "Train_Ready.csv"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOPK = 5


@st.cache_data
def load_classes():
    if not os.path.exists(Config.CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {Config.CSV_PATH}")
    df = pd.read_csv(Config.CSV_PATH)
    if "Label" not in df.columns:
        raise ValueError(f"CSV must contain 'Label' column, got: {list(df.columns)}")
    return sorted(df["Label"].unique().tolist())


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


@st.cache_resource
def load_model(num_classes: int):
    if not os.path.exists(Config.MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {Config.MODEL_WEIGHTS}")

    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=False,
        num_classes=num_classes
    )

    state = torch.load(Config.MODEL_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)

    model.to(Config.DEVICE)
    model.eval()
    return model


def main():
    st.set_page_config(page_title="Skin Disease Recognition (Swin)", layout="wide")
    st.title("🩺 皮肤病图像识别（Swin Transformer）")
    st.caption("上传皮肤病变图像，模型将预测类别并显示 Top-K 概率。")

    topk = st.sidebar.slider("Top-K", 1, 10, Config.TOPK)

    # load
    try:
        classes = load_classes()
        model = load_model(len(classes))
        tfm = get_transform()
    except Exception as e:
        st.error(f"❌ 初始化失败：{e}")
        st.stop()

    st.sidebar.success(f"Device: {Config.DEVICE}")
    st.sidebar.info(f"Classes: {len(classes)}")
    st.sidebar.write("Model:", Config.MODEL_NAME)

    uploaded = st.file_uploader("上传图片（jpg/png）", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("请上传图片开始识别。")
        return

    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("输入图片")
        st.image(img, use_container_width=True)

    x = tfm(img).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    k = min(topk, len(classes))
    top_idx = np.argsort(probs)[::-1][:k]
    top_items = [(classes[i], float(probs[i])) for i in top_idx]
    pred_label, pred_prob = top_items[0]

    with col2:
        st.subheader("预测结果")
        st.markdown(f"### ✅ 预测类别：**{pred_label}**")
        st.markdown(f"**置信度：** `{pred_prob:.4f}`")

        df_show = pd.DataFrame(top_items, columns=["Class", "Probability"])
        st.dataframe(df_show, use_container_width=True)
        st.bar_chart(df_show.set_index("Class"))


if __name__ == "__main__":
    main()


