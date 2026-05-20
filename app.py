import streamlit as st
import cv2
import torch
import tempfile
import os
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="轻眸LiteEye", layout="wide")
st.title("轻眸LiteEye - AI视频语义检索")

st.markdown("""
### 使用说明
1. 上传你的 MP4 视频
2. 点击「开始索引视频」
3. 输入英文关键词搜索（如 dog、cat、sky、person）
4. 点击搜索结果中的时间点即可跳转播放
""")

# 加载模型
@st.cache_resource
def load_model():
    with st.spinner("正在加载AI模型..."):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# 上传视频
uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 保存临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    # 索引视频
    if st.button("开始索引视频"):
        with st.spinner("正在分析视频，请稍候..."):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps)
            
            index = []
            frame_count = 0
            progress_bar = st.progress(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    time_seconds = frame_count / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    inputs = processor(images=frame_rgb, return_tensors="pt")
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                        features = features.cpu().numpy().tolist()
                    
                    index.append({"time": time_seconds, "features": features})
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress_bar.progress(min(frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT), 1.0))
            
            cap.release()
            st.session_state['index'] = index
            st.success(f"✅ 完成！共分析 {len(index)} 个画面")
            os.unlink(video_path)
    
    # 搜索
    query = st.text_input("🔍 搜索画面内容（英文）", placeholder="例如：dog, cat, sky, person, car, beach")
    
    if query and 'index' in st.session_state:
        with st.spinner("搜索中..."):
            results = []
            for frame in st.session_state['index']:
                img_features = torch.tensor(frame["features"])
                inputs = processor(text=[query], return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_features = model.get_text_features(**inputs)
                
                similarity = torch.cosine_similarity(text_features, img_features, dim=1)
                score = similarity[0].item()
                results.append({"time": frame["time"], "score": score})
            
            results.sort(key=lambda x: x["score"], reverse=True)
            top_results = results[:5]
            
            st.subheader("搜索结果")
            for i, r in enumerate(top_results):
                with st.expander(f"🎬 结果 {i+1}: 第 {r['time']:.1f} 秒 (匹配度: {r['score']:.3f})"):
                    st.video(video_path, start_time=r['time'])