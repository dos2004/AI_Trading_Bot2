"""
DeepSeek AI客户端
调用DeepSeek API进行交易决策
"""
import os
import warnings
from typing import Dict, Any

import httpx
from openai import OpenAI


class DeepSeekClient:
    """DeepSeek AI客户端"""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """
        初始化DeepSeek客户端

        Args:
            api_key: DeepSeek API密钥
            model: 模型名称
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.deepseek.com/v1"

        # ✅ FIX: use httpx.Client to support proxies / avoid unsupported kwargs
        http_client = httpx.Client(timeout=120.0)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,  # new style for v1.x
        )

        # 抑制urllib3警告
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    def analyze_and_decide(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        调用AI分析并获取决策
        """
        try:
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的加密货币合约量化交易机器人，负责在严格的规则框架内执行自动化的交易决策。你的核心使命是在控制风险的前提下实现资产稳健增长。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
                **kwargs,
            )

            reasoning_content = None
            message = response.choices[0].message
            content = message.content

            # 推理内容（兼容 deepseek-reasoner）
            if hasattr(message, "reasoning_content"):
                reasoning_content = getattr(message, "reasoning_content", None)
            elif hasattr(response, "reasoning_content"):
                reasoning_content = getattr(response, "reasoning_content", None)
            elif hasattr(response.choices[0], "reasoning_content"):
                reasoning_content = getattr(response.choices[0], "reasoning_content", None)

            return {
                "reasoning_content": reasoning_content,
                "content": content,
                "raw_response": response,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            print(f"❌ DeepSeek API调用失败: {e}")
            raise

    def get_reasoning(self, response: Dict[str, Any]) -> str:
        """获取AI推理过程"""
        return response.get("reasoning_content", "")

    def get_decision_content(self, response: Dict[str, Any]) -> str:
        """获取AI决策内容"""
        return response.get("content", "")

    def calculate_cost(self, response: Dict[str, Any]) -> float:
        """计算API调用成本"""
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        print(f"\n🧠 AI token消耗: prompt_tokens：{prompt_tokens}, completion_tokens: {completion_tokens}")
        # DeepSeek定价示例
        return (prompt_tokens + completion_tokens) / 1000 * 0.002