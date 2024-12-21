import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ollama & LanChain基本
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        page_title="Ask My PDF(s)",
        page_icon="🧐"
    )
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("llama3.2", "gemmma2", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "llama3.2":
        return ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=temperature,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
         # other params...
        )
    elif model == "gemma2":
        return ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=temperature,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
         # other params...
        )
    elif model == "Claude 3.5 Sonnet":
        return ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=temperature,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
         # other params...
        )
    elif model == "Gemini 1.5 Pro":
        return ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=temperature,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
         # other params...
        )


def init_qa_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template("""
    以下の前提知識を用いて、ユーザーからの質問に答えてください。

    ===
    前提知識
    {context}

    ===
    ユーザーからの質問
    {question}
    """)
    retriever = st.session_state.vectorstore.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def page_ask_my_pdf():
    chain = init_qa_chain()

    if query := st.text_input("PDFへの質問を書いてね: ", key="input"):
        st.markdown("## Answer")
        st.write_stream(chain.stream(query))


def main():
    init_page()
    st.title("PDF QA 🧐")
    if "vectorstore" not in st.session_state:
        st.warning("まずは 📄 Upload PDF(s) からPDFファイルをアップロードしてね")
    else:
        page_ask_my_pdf()


if __name__ == '__main__':
    main()
