import streamlit as st
from openai import OpenAI
import os 
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# 新しいClientの初期化方法
chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
# コレクション取得
collection = chroma_client.get_or_create_collection(name="acp_generate")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="acp_generate")

st.set_page_config(page_title="ACP台本生成", layout="wide")
st.title("ACP会話台本生成")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ファイルアップロード
uploaded_files = st.file_uploader(" 書き起こし会話ファイルをアップロード", type="txt", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        embedding = model.encode(text)
        collection.add(documents=[text], embeddings=[embedding], ids=[file.name])
    st.success(" 会話データを登録しました！")


# 条件入力
st.subheader("新しい台本の条件を入力")
condition = st.text_area("例：70代男性／事故による下半身麻痺／在宅療養を希望している")

if st.button("台本を生成"):
    if not condition:
        st.warning("条件を入力してください")
    else:
        # 類似文検索
        query_vec = model.encode(condition)
        results = collection.query(query_embeddings=[query_vec], n_results=2)

        references = "\n---\n".join(doc[0] for doc in results["documents"])

        # プロンプト作成
        prompt = f"""
以下はACP会話の例です：

{references}

---

上記の会話スタイル・トーンを参考にして、以下の条件に基づき、新しいACP会話台本を作成してください。

【条件】
{condition}

【形式】
・話者名（医師、患者）を明記してください。
・15分程度の自然な日本語の対話で構成してください。
"""

        # GPT呼び出し
        with st.spinner("台本を生成中..."):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            script = response.choices[0].message.content
            st.subheader("生成されたACP会話台本")
            st.text(script)
