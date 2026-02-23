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


# =========================
# 1) 配置（按你的路径改）
# =========================
class Config:
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    IMG_SIZE = 224

    # 你的权重文件
    MODEL_WEIGHTS = r"E:/SkinViT_Project/Results/best_model.pth"

    # 读取类别的方式（二选一）
    # A: 用 Train_Ready.csv 自动读取 Label 列（推荐：与你训练一致）
    CSV_PATH = r"E:/SkinViT_Project/Skin_Dataset/Train_Ready.csv"

    # B: 或者你可以准备一个 classes.txt（每行一个类别名），然后改成该路径
    CLASSES_TXT = None  # 例如 r"E:/SkinViT_Project/Results/classes.txt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOPK = 5


# =========================
# 2) 工具函数：加载类别
# =========================
@st.cache_data
def load_classes(csv_path: str, classes_txt: str | None):
    if classes_txt and os.path.exists(classes_txt):
        with open(classes_txt, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError(f"CSV must contain 'Label' column, got columns: {list(df.columns)}")

    classes = sorted(df["Label"].unique().tolist())
    return classes


# =========================
# 3) 推理预处理（与你验证集一致）
# =========================
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# =========================
# 4) 加载模型
# =========================
@st.cache_resource
def load_model(num_classes: int):
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=False,          # 推理用你自己的权重，不用预训练权重
        num_classes=num_classes
    )

    if not os.path.exists(Config.MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {Config.MODEL_WEIGHTS}")

    state = torch.load(Config.MODEL_WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.to(Config.DEVICE)
    model.eval()
    return model


# =========================
# 5) Grad-CAM（简单版）
#   - 对 Swin/ViT CAM 不是完美，但能做一个“可用的热力图”
# =========================
def compute_cam_swin(model, input_tensor, class_idx: int):
    """
    返回 heatmap (H, W) in [0,1]
    说明：对 timm 的 Swin，我们尝试hook最后一个stage输出特征。
    """
    feats = {}
    grads = {}

    # timm Swin 通常有 model.layers[-1].blocks[-1] 或 model.layers[-1] 可hook
    # 为了稳健：优先 hook model.layers[-1]
    target_module = None
    if hasattr(model, "layers") and len(model.layers) > 0:
        target_module = model.layers[-1]
    else:
        # 兜底：hook 倒数第二层模块
        modules = list(model.modules())
        target_module = modules[-2]

    def fwd_hook(module, inp, out):
        feats["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        grads["value"] = grad_out[0]

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[:, class_idx].sum()
    score.backward()

    h1.remove()
    h2.remove()

    if "value" not in feats or "value" not in grads:
        return None

    f = feats["value"]
    g = grads["value"]

    # Swin的特征可能是 (B, H, W, C) 或 (B, N, C) 或 (B, C, H, W)
    # 做一些兼容处理
    if f.dim() == 4 and f.shape[1] != 3 and f.shape[1] < 32 and f.shape[-1] > 32:
        # 可能是 (B, H, W, C)
        # 转成 (B, C, H, W)
        f = f.permute(0, 3, 1, 2).contiguous()
        g = g.permute(0, 3, 1, 2).contiguous()
    elif f.dim() == 3:
        # (B, N, C) -> 估算成 sqrt(N) 的 2D
        B, N, C = f.shape
        side = int(np.sqrt(N))
        if side * side == N:
            f = f.permute(0, 2, 1).contiguous().view(B, C, side, side)
            g = g.permute(0, 2, 1).contiguous().view(B, C, side, side)
        else:
            return None

    if f.dim() != 4:
        return None

    # Grad-CAM: channel-wise weight = global avg pool of grads
    weights = g.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
    cam = (weights * f).sum(dim=1, keepdim=False)  # (B,H,W)
    cam = F.relu(cam)

    # normalize to [0,1]
    cam = cam[0]
    cam = cam - cam.min()
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def overlay_heatmap_on_image(img_pil: Image.Image, heatmap: np.ndarray, alpha=0.45):
    """
    把 heatmap 叠加到原图上，返回叠加后的 PIL
    """
    import matplotlib.cm as cm

    img = img_pil.convert("RGB")
    w, h = img.size
    hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    hm_np = np.array(hm) / 255.0
    colored = cm.jet(hm_np)[:, :, :3]  # (H,W,3)
    colored = (colored * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored)

    blended = Image.blend(img, colored_img, alpha=alpha)
    return blended


# =========================
# 6) Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Skin Disease Recognition (Swin)", layout="wide")

    st.title("🩺 Skin Disease Image Recognition (Swin Transformer)")
    st.caption("Upload a skin lesion image, the model will predict the class and show Top-K probabilities.")

    # Sidebar
    st.sidebar.header("Settings")
    topk = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=Config.TOPK)
    show_cam = st.sidebar.checkbox("Show heatmap (Grad-CAM)", value=True)
    alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, 0.45, 0.05)

    # Load classes & model
    try:
        classes = load_classes(Config.CSV_PATH, Config.CLASSES_TXT)
        model = load_model(num_classes=len(classes))
        tfm = get_transform()
    except Exception as e:
        st.error(f"❌ Failed to load model/classes: {e}")
        st.stop()

    st.sidebar.success(f"Device: {Config.DEVICE}")
    st.sidebar.info(f"Classes: {len(classes)}")
    st.sidebar.write("Model:", Config.MODEL_NAME)

    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("Please upload an image to start.")
        return

    # Read image
    image_bytes = uploaded.read()
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(img_pil, use_container_width=True)

    # Prepare input tensor
    x = tfm(img_pil).unsqueeze(0).to(Config.DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # Top-k
    k = min(topk, len(classes))
    top_idx = np.argsort(probs)[::-1][:k]
    top_items = [(classes[i], float(probs[i])) for i in top_idx]

    pred_label, pred_prob = top_items[0]

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### ✅ Predicted: **{pred_label}**")
        st.markdown(f"**Confidence:** `{pred_prob:.4f}`")

        st.write("Top-K probabilities:")
        df_show = pd.DataFrame(top_items, columns=["Class", "Probability"])
        st.dataframe(df_show, use_container_width=True)

        st.bar_chart(df_show.set_index("Class"))

    # Heatmap
    if show_cam:
        st.subheader("Heatmap (Grad-CAM)")
        class_idx = classes.index(pred_label)
        cam = compute_cam_swin(model, x, class_idx=class_idx)
        if cam is None:
            st.warning("Heatmap generation failed for this model structure. (Swin/ViT CAM may require custom hooks.)")
        else:
            overlay = overlay_heatmap_on_image(img_pil, cam, alpha=alpha)
            st.image(overlay, caption="CAM overlay", use_container_width=True)


if __name__ == "__main__":
    main()
