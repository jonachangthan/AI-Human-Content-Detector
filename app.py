import streamlit as st
from transformers import pipeline

# 設定頁面標題與圖示
st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖")

st.title("🤖 AI 文本偵測器")
st.markdown("請輸入一段英文文章，模型將判斷這段文字是由 **AI 生成** 還是 **人類撰寫**。")

# 側邊欄說明
with st.sidebar:
    st.header("關於模型")
    st.info("本工具使用 Hugging Face 的 Transformers 庫，載入 `roberta-base-openai-detector` 模型進行推論。")
    st.markdown("---")
    st.write("Created for HW5 - Advanced Topic")

# 1. 載入模型
@st.cache_resource
def load_model():
    classifier = pipeline("text-classification", model="roberta-base-openai-detector")
    return classifier

try:
    with st.spinner('正在載入 AI 偵測模型...（首次執行需下載模型，約需 1-2 分鐘）'):
        pipe = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"模型載入失敗: {e}")
    model_loaded = False

# 2. 使用者介面 - 文字輸入
user_input = st.text_area("在此貼上要檢測的文本 (目前模型對英文支援度最佳):", height=200)

# 3. 執行偵測
if st.button("開始分析 🚀") and model_loaded:
    if not user_input.strip():
        st.warning("請先輸入文字！")
    else:
        with st.spinner('AI 正在讀取並分析特徵...'):
            # --- 修正開始 ---
            # 加入 truncation=True 與 max_length=512 解決長度報錯問題
            results = pipe(user_input, top_k=None, truncation=True, max_length=512)
            # --- 修正結束 ---
            
            # 整理數據
            scores = {item['label']: item['score'] for item in results}
            
            # 原始模型標籤定義： 'Real' = Human, 'Fake' = AI
            ai_score = scores.get('Fake', 0.0)
            human_score = scores.get('Real', 0.0)

        # 4. 顯示結果
        st.subheader("📊 分析結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="AI 生成機率", value=f"{ai_score:.2%}")
            st.progress(ai_score)
            
        with col2:
            st.metric(label="人類撰寫機率", value=f"{human_score:.2%}")
            st.progress(human_score, "primary")

        st.divider()
        if ai_score > 0.8:
            st.error("🚨 判定結果：這極大機率是由 **AI (如 ChatGPT)** 生成的內容。")
        elif ai_score > 0.5:
            st.warning("⚠️ 判定結果：這段文字帶有 **AI 生成特徵**，可能是混合撰寫。")
        else:
            st.success("✅ 判定結果：這看起來像是 **人類 (Human)** 撰寫的內容。")

        with st.expander("查看原始數據"):
            st.json(results)
