import fitz  # PyMuPDF
import streamlit as st

# Ollama & LanChain基本
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Upload PDF(s)",
        page_icon="📄"
    )
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear DB", key="clear")
    if clear_button and "vectorstore" in st.session_state:
        del st.session_state.vectorstore


def get_pdf_text():
    # file_uploader でPDFをアップロードする
    # (file_uploaderの詳細な説明は第6章をご参照ください)
    pdf_file = st.file_uploader(
        label='Upload your PDF 😇',
        type='pdf'  # PDFファイルのみアップロード可
    )
    if pdf_file:
        pdf_text = ""
        with st.spinner("Loading PDF ..."):
            # PyMuPDFでPDFを読み取る
            # (詳細な説明はライブラリの公式ページなどをご参照ください)
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()

        # RecursiveCharacterTextSplitter は、テキストを特定のルールに従って効率的に分割するためのツール
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            # チャンクが完全に独立した形で分割されると、分割点付近の文脈（前後関係）が失われ、モデルやシステムが適切に理解できなくなる可能性があるため、overlapを設ける
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(pdf_text)
    else:
        return None


def build_vector_store(pdf_text):
    with st.spinner("Saving to vector store ..."):
        if 'vectorstore' in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            # PDFから抽出したテキスト (pdf_text) をベクトル化し、FAISS（Facebook AI Similarity Search）という効率的な類似検索ライブラリを使ってベクトルストアを作成するプロセス
            # ベクトルDBの初期化と文書の追加を同時に行う
            # LangChain の Document Loader を利用した場合は `from_documents` にする
            st.session_state.vectorstore = FAISS.from_texts(
                pdf_text,
                OpenAIEmbeddings(model="text-embedding-3-small")
            )

            # FAISSのデフォルト設定はL2距離となっている
            # コサイン類似度にしたい場合は以下のようにする
            # from langchain_community.vectorstores.utils import DistanceStrategy
            # st.session_state.vectorstore = FAISS.from_texts(
            #     pdf_text,
            #     OpenAIEmbeddings(model="text-embedding-3-small"),
            #     distance_strategy=DistanceStrategy.COSINE
            # )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)


def main():
    init_page()
    page_pdf_upload_and_build_vector_db()


if __name__ == '__main__':
    main()
