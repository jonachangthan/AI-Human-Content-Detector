import streamlit as st
from transformers import pipeline

# 設定頁面標題與圖示
st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖")

st.title("🤖 AI 文本偵測器")
st.markdown("請輸入一段英文文章（建議長度 50 字以上），模型將判斷這段文字是由 **AI 生成** 還是 **人類撰寫**。")

# 側邊欄說明
with st.sidebar:
    st.header("關於模型")
    st.info("本工具使用 Hugging Face 的 Transformers 庫，載入 `roberta-base-openai-detector` 模型進行推論。")
    st.markdown("---")
    st.write("Created for HW5 - Advanced Topic")

# 1. 載入模型 (使用 @st.cache_resource 避免每次重整都重新下載/載入)
@st.cache_resource
def load_model():
    # 這裡使用 pipeline 自動下載並載入預訓練模型
    # 注意：第一次執行會下載約 500MB 的模型權重，請耐心等候
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
        # 模型通常有長度限制 (512 tokens)，這裡簡單截斷過長的輸入以防報錯
        # 實際產品應用需做分段處理 (Chunking)
        truncated_input = user_input[:2000] 
        
        with st.spinner('AI 正在讀取並分析特徵...'):
            # 獲取預測結果，return_all_scores=True 讓我們同時拿到 Real 和 Fake 的機率
            # 注意：新版 transformers pipeline 參數可能是 top_k=None
            results = pipe(truncated_input, top_k=None)
            
            # results 結構通常是 [[{'label': 'Real', 'score': 0.9}, {'label': 'Fake', 'score': 0.1}]]
            # 整理數據
            scores = {item['label']: item['score'] for item in results}
            
            # 原始模型標籤定義：
            # 'Real' = Human (人類)
            # 'Fake' = AI (生成)
            ai_score = scores.get('Fake', 0.0)
            human_score = scores.get('Real', 0.0)

        # 4. 顯示結果
        st.subheader("📊 分析結果")
        
        # 建立兩欄佈局
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="AI 生成機率", value=f"{ai_score:.2%}")
            st.progress(ai_score)
            
        with col2:
            st.metric(label="人類撰寫機率", value=f"{human_score:.2%}")
            st.progress(human_score, "primary") # primary color usually indicates 'good' or 'base'

        # 根據結果顯示結論
        st.divider()
        if ai_score > 0.8:
            st.error("🚨 判定結果：這極大機率是由 **AI (如 ChatGPT)** 生成的內容。")
        elif ai_score > 0.5:
            st.warning("⚠️ 判定結果：這段文字帶有 **AI 生成特徵**，可能是混合撰寫。")
        else:
            st.success("✅ 判定結果：這看起來像是 **人類 (Human)** 撰寫的內容。")

        # 顯示原始數據 (Debugging用途，也可作為作業的詳細輸出)
        with st.expander("查看原始數據"):
            st.json(results)