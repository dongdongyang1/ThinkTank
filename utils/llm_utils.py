from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

_llm_client_cache = {}

def get_llm_client(model : str | None = None,json_mode : bool = False)->ChatOpenAI:
    """
     获取 LangChain ChatOpenAI 客户端实例
     - model: 允许不同节点使用不同模型
     - json_mode: True 时要求输出 JSON
     """
    m = model or lm_config.llm_model
    #同一个模型，开启 / 关闭 json 格式化是两个独立的 LLM 实例，不能共用缓存，必须分开存。
    key = (m,json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    extra_body = {"enable_thinking":False}
    #定义一个空字典 model_kwargs，用来存放传给大模型的额外参数
    #response_format={"type": "json_object"} 是 ChatOpenAI 的官方参数
    model_kwargs : dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type":"json_object"}
    client = ChatOpenAI(
        model = m,
        temperature = lm_config.llm_temperature,
        api_key = lm_config.api_key,
        base_url = lm_config.base_url,
        extra_body = extra_body,
        model_kwargs = model_kwargs,
        timeout=60,  # 单次请求超时60秒
        max_retries=2,  # 失败自动重试2次

    )
    _llm_client_cache[key]=client
    return client
