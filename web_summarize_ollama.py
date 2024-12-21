# 5章アプリ
import os
import traceback
import tiktoken
import streamlit as st

# Ollama & LanChain基本
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import requests                     # PythonでHTTPリクエストを簡単に送信するためのライブラリ
from bs4 import BeautifulSoup
from urllib.parse import urlparse   # Python の標準ライブラリで、URL の解析 (parsing) や操作を簡単に行うためのモジュール

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

SUMMARIZE_PROMPT = """以下のコンテンツについて、内容を300文字程度でわかりやすく要約してください。

========

{content}

========

日本語で書いてね！
"""


def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("llama3.2", "gemmma2", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "llama3.2":
        st.session_state.model_name = "llama3.2"
        return ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=0,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
    # other params...
        )   
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o"
        )
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature,
            model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-1.5-pro-latest"
        )


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", SUMMARIZE_PROMPT),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def validate_url(url):
    """ URLが有効かどうかを判定する関数 """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    try:
        with st.spinner("Fetching Website ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # なるべく本文の可能性が高い要素を取得する
            if soup.main:                   # soup.main: HTML の <main> 要素を取得します。
                return soup.main.get_text()
            elif soup.article:              # soup.article: <main> 要素が存在しない場合、代わりに <article> 要素を確認します。
                return soup.article.get_text()
            else:                           # soup.body: <main> も <article> も存在しない場合、HTML全体の <body> 要素を取得します。
                return soup.body.get_text()
    except:
        st.write(traceback.format_exc())  # エラーが発生した場合はエラー内容を表示
        return None


def main():
    init_page()
    chain = init_chain()

    # ユーザーの入力を監視
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

    # コストを表示する場合は第3章と同じ実装を追加してください
    # calc_and_display_costs()


if __name__ == '__main__':
    main()
