# AI / Human Text Detector (AI 文章偵測器)

## 📖 專案簡介 (Project Description)
這是一個基於 NLP 深度學習模型的 Web 應用程式，旨在檢測輸入的英文文本是由 **人工智慧 (如 ChatGPT)** 生成的，還是由 **人類 (Human)** 撰寫的。

本專案使用了 Hugging Face 的 Transformers 函式庫與 Streamlit 框架進行開發。

## 🚀 功能特色 (Features)
* **即時偵測**：輸入文本後立即進行推論分析。
* **深度學習模型**：使用 `roberta-base-openai-detector` 模型進行特徵辨識。
* **視覺化結果**：透過 Streamlit 顯示 AI 與人類的機率佔比條形圖。
* **簡易 UI**：直觀的網頁操作介面。

## 🛠️ 安裝與執行 (Installation & Usage)

### 1. 複製專案 (Clone Repository)
```bash
git clone https://github.com/jonachangthan/AI-Human-Text-Detector.git
```


### 2. 環境設定 (Environment Setup)
建議使用 Python 虛擬環境以保持開發環境乾淨：
```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate
```


### 3. 安裝依賴套件 (Install Dependencies)
請執行以下指令安裝 requirements.txt 中列出的必要套件：
```bash
pip install -r requirements.txt
```

### 4. 啟動應用程式 (Run App)
環境準備完成後，執行以下指令啟動 Streamlit：
```bash
streamlit run app.py
```

#### 啟動成功後，瀏覽器將自動開啟 http://localhost:8501。

#### 注意：首次執行時，程式會自動從 Hugging Face 下載約 500MB 的模型權重，請耐心等候。

## 📂 檔案結構 (File Structure)
app.py: Streamlit 主程式與模型推論邏輯。

requirements.txt: 專案所需的 Python 套件清單 (包含 streamlit, transformers, torch)。

README.md: 專案說明文件。

## 🤖 技術棧 (Tech Stack)
Streamlit

Hugging Face Transformers

PyTorch
