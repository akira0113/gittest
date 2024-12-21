# ２章アプリ
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain           #未使用
from langchain_ollama.llms import OllamaLLM     #未使用
from langchain_openai import ChatOpenAI         #未使用

###### API＿KEYなどを環境変数に入れている場合の読みだす処理　dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def main():
    st.set_page_config(
        page_title="ローカルLLMとチャット:llama3.2",
        page_icon="🤗"
    )
    st.header("ローカルLLMとチャット:llama3.2")

    # チャット履歴の初期化: message_history がなければ作成
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            # System Prompt を設定 ('system' はSystem Promptを意味する)
            ("system", "関西弁で答えてください。")
        ]

    # 1. LLMに質問を与えて回答を取り出す(パースする)処理を作成 (1.-4.の処理)
    #llm = ChatOpenAI(temperature=0)
    #llm = OllamaLLM(model="llama3.2")
    llm = ChatOllama(
        model="llama3.2",   #model（str）: 使用するOllamaモデルの名称を指定します。例: "llama3.2" Ollamaで利用可能なモデルの名前を指定します。モデルは事前にダウンロードしておく必要があります。
        temperature=0,      #temperature（float）: サンプリング時の温度を設定します。この値を調整することで、生成されるテキストの多様性を制御できます。値が低いほど出力はより決定的（同じ入力に対して同じ出力を生成）になり、高いほど出力に多様性が増します。
    # other params...
    )

    # 2. ユーザーの質問を受け取り、ChatGPTに渡すためのテンプレートを作成
    #    テンプレートには過去のチャット履歴を含めるように設定
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")  # ここにあとでユーザーの入力が入る
    ])

    # 3. ChatGPTの返答をパースするための処理を呼び出し
    output_parser = StrOutputParser()

    # 4. ユーザーの質問をChatGPTに渡し、返答を取り出す連続的な処理(chain)を作成
    #    各要素を | (パイプ) でつなげて連続的な処理を作成するのがLCELの特徴
    chain = prompt | llm | output_parser

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        with st.spinner("LLM is typing ..."):
            response = chain.invoke({"user_input": user_input})

        # ユーザーの質問を履歴に追加 ('user' はユーザーの質問を意味する)
        st.session_state.message_history.append(("user", user_input))

        # ChatGPTの回答を履歴に追加 ('assistant' はChatGPTの回答を意味する)
        st.session_state.message_history.append(("ai", response))

    # チャット履歴の表示
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == '__main__':
    main()
