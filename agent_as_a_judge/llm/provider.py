import warnings
import time
import os
from dotenv import load_dotenv
import yaml

__all__ = ["LLM"]

message_separator = "\n\n----------\n\n"

class LLM:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        api_version=None,
        num_retries=3,
        retry_min_wait=1,
        retry_max_wait=10,
        llm_timeout=30,
        llm_temperature=0.7,
        llm_top_p=0.9,
        custom_llm_provider=None,
        provider="qwen",
        max_input_tokens=4096,
        max_output_tokens=2048,
        cost=None,
        **kwargs
    ):
        from agent_as_a_judge.llm.cost import Cost
        self.cost = Cost()
        self.model_name = model or os.getenv("DEFAULT_LLM")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.api_version = api_version
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.llm_timeout = llm_timeout
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.num_retries = num_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.custom_llm_provider = custom_llm_provider
        self.provider = provider or custom_llm_provider or "qwen"
        self.kwargs = kwargs
        self.model_info = None
        self._initialize_completion_function()

    def _initialize_completion_function(self):
        if self.provider == "qwen":
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            def wrapper(messages, **kwargs):
                # 避免重复传递参数
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    max_tokens=self.max_output_tokens,
                    **kwargs
                )
                result = completion.model_dump()
                message_back = result["choices"][0]["message"]["content"]
                return result, message_back
            self._completion = wrapper
        elif self.provider == "baichuan":
            import requests
            def wrapper(messages, **kwargs):
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                url = self.base_url or "https://api.baichuan-ai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.llm_temperature,
                    "top_p": self.llm_top_p,
                    "max_tokens": self.max_output_tokens,
                }
                data.update(kwargs)
                resp = requests.post(url, headers=headers, json=data, timeout=self.llm_timeout)
                resp.raise_for_status()
                result = resp.json()
                message_back = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result, message_back
            self._completion = wrapper
        elif self.provider == "glm":
            import requests
            def wrapper(messages, **kwargs):
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                url = self.base_url or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.llm_temperature,
                    "top_p": self.llm_top_p,
                    "max_tokens": self.max_output_tokens,
                }
                data.update(kwargs)
                resp = requests.post(url, headers=headers, json=data, timeout=self.llm_timeout)
                resp.raise_for_status()
                result = resp.json()
                message_back = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result, message_back
            self._completion = wrapper
        elif self.provider == "minimax":
            import requests
            def wrapper(messages, **kwargs):
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                url = self.base_url or "https://api.minimax.chat/v1/text/chatcompletion"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.llm_temperature,
                    "top_p": self.llm_top_p,
                    "max_tokens": self.max_output_tokens,
                }
                data.update(kwargs)
                resp = requests.post(url, headers=headers, json=data, timeout=self.llm_timeout)
                resp.raise_for_status()
                result = resp.json()
                message_back = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result, message_back
            self._completion = wrapper
        elif self.provider == "spark":
            import requests
            def wrapper(messages, **kwargs):
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                url = self.base_url or "https://spark-api.xf-yun.com/v3.5/chat"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.llm_temperature,
                    "top_p": self.llm_top_p,
                    "max_tokens": self.max_output_tokens,
                }
                data.update(kwargs)
                resp = requests.post(url, headers=headers, json=data, timeout=self.llm_timeout)
                resp.raise_for_status()
                result = resp.json()
                message_back = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result, message_back
            self._completion = wrapper
        elif self.provider == "custom_http":
            import requests
            def wrapper(messages, **kwargs):
                for k in ["temperature", "top_p", "max_tokens"]:
                    kwargs.pop(k, None)
                url = self.base_url
                headers = {"Authorization": f"Bearer {self.api_key}"}
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.llm_temperature,
                    "top_p": self.llm_top_p,
                    "max_tokens": self.max_output_tokens,
                }
                data.update(kwargs)
                resp = requests.post(url, headers=headers, json=data, timeout=self.llm_timeout)
                resp.raise_for_status()
                result = resp.json()
                message_back = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return result, message_back
            self._completion = wrapper
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @property
    def completion(self):
        return self._completion

    def _llm_inference(self, messages: list) -> dict:
        start_time = time.time()
        response, cost, accumulated_cost = self.do_completion(messages=messages, temperature=0.0)
        inference_time = time.time() - start_time
        llm_response = response["choices"][0]["message"]["content"] if "choices" in response else ""
        input_token = response.get("usage", {}).get("prompt_tokens", 0)
        output_token = response.get("usage", {}).get("completion_tokens", 0)
        return {
            "llm_response": llm_response,
            "input_tokens": input_token,
            "output_tokens": output_token,
            "cost": cost if "cost" in locals() else 0,
            "accumulated_cost": accumulated_cost if "accumulated_cost" in locals() else 0,
            "inference_time": inference_time,
        }

    def do_completion(self, *args, **kwargs):
        resp, msg = self._completion(*args, **kwargs)
        cur_cost, accumulated_cost = 0, 0
        return resp, cur_cost, accumulated_cost

    def post_completion(self, response: dict):
        return 0.0, 0.0

    def get_token_count(self, messages):
        return 0

    def is_local(self):
        if self.base_url:
            return any(
                substring in self.base_url
                for substring in ["localhost", "127.0.0.1", "0.0.0.0"]
            )
        if self.model_name and self.model_name.startswith("ollama"):
            return True
        return False

    def completion_cost(self, response):
        return 0.0

    def __str__(self):
        return f"LLM(model={self.model_name}, base_url={self.base_url})"

    def __repr__(self):
        return str(self)

    def do_multimodal_completion(self, text, image_path):
        messages = self.prepare_messages(text, image_path=image_path)
        response, cur_cost, accumulated_cost = self.do_completion(messages=messages)
        return response, cur_cost, accumulated_cost

    @staticmethod
    def encode_image(image_path):
        import base64

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_messages(self, text, image_path=None):
        messages = [{"role": "user", "content": text}]
        if image_path:
            base64_image = self.encode_image(image_path)
            messages[0]["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64_image},
                },
            ]
        return messages

    @classmethod
    def from_config(
        cls,
        model_name=None,
        api_key=None,
        base_url=None,
        custom_llm_provider=None,
        provider=None,
        config_path="llm_config.yaml"
    ):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_key = model_name or config.get("default")
            params = config["models"].get(model_key, {})
            # 环境变量替换
            for k, v in params.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    env_key = v[2:-1]
                    params[k] = os.getenv(env_key, "")
            # 允许外部参数覆盖配置文件
            if api_key:
                params["api_key"] = api_key
            if base_url:
                params["base_url"] = base_url
            if custom_llm_provider:
                params["custom_llm_provider"] = custom_llm_provider
            if provider:
                params["provider"] = provider
            return cls(model=model_key, **params)
        except Exception:
            # 兜底：环境变量
            return cls(
                model=model_name or os.getenv("DEFAULT_LLM"),
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                custom_llm_provider=custom_llm_provider,
                provider=provider or os.getenv("LLM_PROVIDER", "qwen")
            )


if __name__ == "__main__":
    load_dotenv()

    model_name = "qwen-plus"
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    llm_instance = LLM(model=model_name, api_key=api_key, base_url=base_url)

    image_path = "/Users/zhugem/Desktop/DevAI/studio/workspace/sample/results/prediction_interactive.png"

    for i in range(1):

        multimodal_response = llm_instance.do_multimodal_completion(
            "What's in this image?", image_path
        )
        print(multimodal_response)

