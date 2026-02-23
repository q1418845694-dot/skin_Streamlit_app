import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="皮肤病智能识别 - Swin Transformer",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 皮肤病智能识别系统 (Swin Transformer)")
st.markdown("上传皮肤镜图像，模型将预测其所属的病变类别。")

# -------------------- 全局缓存 --------------------
@st.cache_resource
def load_model(model_path, num_classes, device):
    """加载 Swin Transformer 模型"""
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_data
def load_class_names_from_csv(csv_file):
    """从训练时使用的 CSV 文件中提取类别名称（与训练时顺序一致）"""
    df = pd.read_csv(csv_file)
    classes = sorted(list(df['Label'].unique()))   # 训练时也是 sorted
    return classes

# -------------------- 固定加载本地模型与类别 --------------------

DEFAULT_MODEL_PATH = "best_model.pth"
DEFAULT_CSV_PATH   = "Train_Ready.csv"

# 检查文件是否存在
if not os.path.exists(DEFAULT_MODEL_PATH):
    st.error("❌ 未找到模型文件 best_model.pth")
    st.stop()

if not os.path.exists(DEFAULT_CSV_PATH):
    st.error("❌ 未找到 Train_Ready.csv 文件")
    st.stop()

# 读取类别
class_names = load_class_names_from_csv(DEFAULT_CSV_PATH)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(DEFAULT_MODEL_PATH, len(class_names), device)

st.sidebar.markdown("### ⚙️ 系统信息")
st.sidebar.markdown(f"类别数量: {len(class_names)}")
st.sidebar.markdown(f"运行设备: `{device}`")

# -------------------- 图像预处理 --------------------
# 严格对齐验证集预处理
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- 主界面：图像上传与预测 --------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_img = st.file_uploader(
        "📤 上传皮肤镜图像",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="支持常见图像格式"
    )

    if uploaded_img is not None:
        # 显示原图
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="原始图像", use_column_width=True)

if uploaded_img is not None and model is not None:
    with col2:
        st.subheader("🔍 预测结果")

        # 预处理
        input_tensor = val_transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        # 转换为 numpy
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx  = top5_idx.cpu().numpy()[0]
        top5_labels = [class_names[i] for i in top5_idx]

        # 显示 Top-1 结果
        st.markdown(f"### 🥇 预测: **{top5_labels[0]}**")
        st.markdown(f"置信度: **{top5_prob[0]:.2%}**")

        # 显示 Top-5 条形图
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("Blues_d", len(top5_prob))
        y_pos = np.arange(len(top5_labels))
        ax.barh(y_pos, top5_prob, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("置信度")
        ax.set_title("Top-5 预测")
        ax.set_xlim(0, 1)
        for i, (prob, label) in enumerate(zip(top5_prob, top5_labels)):
            ax.text(prob + 0.01, i, f"{prob:.2%}", va='center')

        st.pyplot(fig)

        # 可选：显示所有类别置信度的迷你条形图（折叠）
        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probabilities.cpu().numpy()[0]
            # 按置信度降序排序
            sorted_indices = np.argsort(all_prob)[::-1]
            sorted_labels = [class_names[i] for i in sorted_indices[:10]]  # 只显示前10
            sorted_probs = all_prob[sorted_indices[:10]]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(np.arange(len(sorted_labels)), sorted_probs, color='lightcoral')
            ax2.set_yticks(np.arange(len(sorted_labels)))
            ax2.set_yticklabels(sorted_labels, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("置信度")
            ax2.set_title("Top-10 类别")
            ax2.set_xlim(0, 1)
            st.pyplot(fig2)

else:
    with col2:
        st.info("👈 请先上传一张皮肤图像")

# -------------------- 页脚说明 --------------------
st.markdown("---")
st.markdown("""
**使用说明**  
请上传您的发病部位的清晰图片，系统将为您诊断出最可能的皮肤病类型  
""")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="皮肤病智能识别 - Swin Transformer",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 皮肤病智能识别系统 (Swin Transformer)")
st.markdown("上传皮肤镜图像，模型将预测其所属的病变类别。")

# -------------------- 硬编码路径（请根据实际情况修改） --------------------
MODEL_PATH = "best_model.pth"          # 模型权重文件路径
CSV_PATH   = "Train_Ready.csv"         # 训练集 CSV 文件路径（用于读取类别名称）

# -------------------- 检查文件是否存在 --------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 模型文件不存在：{MODEL_PATH}\n请检查路径或修改代码中的 MODEL_PATH 变量。")
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error(f"❌ CSV 文件不存在：{CSV_PATH}\n请检查路径或修改代码中的 CSV_PATH 变量。")
    st.stop()

# -------------------- 全局缓存 --------------------
@st.cache_resource
def load_model(model_path, num_classes, device):
    """加载 Swin Transformer 模型"""
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_data
def load_class_names_from_csv(csv_file):
    """从训练时使用的 CSV 文件中提取类别名称（与训练时顺序一致）"""
    df = pd.read_csv(csv_file)
    classes = sorted(list(df['Label'].unique()))   # 训练时也是 sorted
    return classes

# -------------------- 加载类别名称 --------------------
class_names = load_class_names_from_csv(CSV_PATH)
st.sidebar.success(f"✅ 已加载 {len(class_names)} 个类别")

# -------------------- 加载模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, len(class_names), device)
st.sidebar.success(f"✅ 模型加载成功 (设备: {device})")

# -------------------- 图像预处理 --------------------
# 严格对齐验证集预处理
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- 主界面：图像上传与预测 --------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_img = st.file_uploader(
        "📤 上传皮肤镜图像",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="支持常见图像格式"
    )

    if uploaded_img is not None:
        # 显示原图
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="原始图像", use_column_width=True)

if uploaded_img is not None and model is not None:
    with col2:
        st.subheader("🔍 预测结果")

        # 预处理
        input_tensor = val_transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        # 转换为 numpy
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx  = top5_idx.cpu().numpy()[0]
        top5_labels = [class_names[i] for i in top5_idx]

        # 显示 Top-1 结果
        st.markdown(f"### 🥇 预测: **{top5_labels[0]}**")
        st.markdown(f"置信度: **{top5_prob[0]:.2%}**")

        # 显示 Top-5 条形图
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = sns.color_palette("Blues_d", len(top5_prob))
        y_pos = np.arange(len(top5_labels))
        ax.barh(y_pos, top5_prob, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("置信度")
        ax.set_title("Top-5 预测")
        ax.set_xlim(0, 1)
        for i, (prob, label) in enumerate(zip(top5_prob, top5_labels)):
            ax.text(prob + 0.01, i, f"{prob:.2%}", va='center')

        st.pyplot(fig)

        # 可选：显示所有类别置信度的迷你条形图（折叠）
        with st.expander("📊 查看所有类别的置信度分布"):
            all_prob = probabilities.cpu().numpy()[0]
            # 按置信度降序排序
            sorted_indices = np.argsort(all_prob)[::-1]
            sorted_labels = [class_names[i] for i in sorted_indices[:10]]  # 只显示前10
            sorted_probs = all_prob[sorted_indices[:10]]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(np.arange(len(sorted_labels)), sorted_probs, color='lightcoral')
            ax2.set_yticks(np.arange(len(sorted_labels)))
            ax2.set_yticklabels(sorted_labels, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("置信度")
            ax2.set_title("Top-10 类别")
            ax2.set_xlim(0, 1)
            st.pyplot(fig2)

else:
    with col2:
        st.info("👈 请先上传一张皮肤图像")

# -------------------- 页脚说明 --------------------
st.markdown("---")
st.markdown("""
**使用说明**  
- 本应用已预加载模型和类别信息，无需额外配置。  
- 上传皮肤镜图像后自动预测并显示结果。  
""")

