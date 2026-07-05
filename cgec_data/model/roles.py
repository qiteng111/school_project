import os
import sys 
import datetime
import asyncio


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class gen_cov:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    def create_prefix_cache(self, prefix_content):
        """
        创建前缀缓存。
        注意：prefix_content 最好超过 256 tokens，否则可能无法创建 prefix cache。
        """
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": prefix_content
                }
            ],
            caching={
                "type": "enabled",
                "prefix": True
            },
            thinking={
                "type": "disabled"
            },
        )
        return response.id

    def pre_api(self, previous_response_id, user_message, temperature=0.7):
        """
        基于已经创建好的 previous_response_id 继续提问。
        """
        response = self.client.responses.create(
            model=self.model,
            previous_response_id=previous_response_id,
            input=[
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            thinking={
                "type": "disabled"
            },
            temperature=temperature,
            # max_tokens=16000,
        )

        # 不同 SDK 版本返回结构可能略有差异
        # 优先尝试 output_text
        if hasattr(response, "output_text"):
            return response.output_text

        # 兼容部分 responses 返回格式
        try:
            return response.output[0].content[0].text
        except Exception:
            return str(response)
        

# class gen_cov:
#     def __init__(self,model,client):
#         self.model = model
#         self.client = client


#     # 创建前缀缓存
#     def create_prefix_cache(self,prefix_messages):
#         response = self.client.context.create(
#             model=self.model,
#             mode="common_prefix",
#             messages=prefix_messages,
#             ttl=datetime.timedelta(days=7) 
#         )
#         return response.id
    
#     # 调用 API
#     def pre_api(self,context_id, user_message,temperature=0.7):
#         chat_response = self.client.context.completions.create(
#             context_id=context_id,
#             model=self.model,
#             messages=[{"role": "user", "content": user_message}],
#             stream=False,
#             temperature=temperature,
#             max_tokens=16000
#         )
#         return chat_response.choices[0].message.content
    
#     def pre_api_pro(self, context_id, user_message, timeout=60):
#         try:
#             chat_response = self.client.context.completions.create(
#                 context_id=context_id,
#                 model=self.model,
#                 messages=[{"role": "user", "content": user_message}],
#                 stream=False,
#                 timeout=timeout,  # ⏱ 设置 API 超时时间
#             )
#             return chat_response.choices[0].message.content
#         except Exception as e:
#             print(f"⚠️ pre_api 调用失败: {e}")
#             raise 

    

    
import asyncio
import httpx

class AsyncGenCov:
    def __init__(self, api_base_url, model, api_key):
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 创建前缀缓存
    def create_prefix_cache(self,prefix_messages):
        response = self.client.context.create(
            model=self.model,
            mode="common_prefix",
            messages=prefix_messages,
            ttl=datetime.timedelta(days=1)  # 缓存 1 天
        )
        return response.id
    
    # 创建前缀缓存
    async def create_prefix_cache(self, prefix_messages, ttl_days=1):
        url = f"{self.api_base_url}/context/create"
        payload = {
            "model": self.model,
            "mode": "common_prefix",
            "messages": prefix_messages,
            "ttl": ttl_days * 86400  # 转换成秒
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()["id"]

    # 异步调用 API
    async def pre_api(self, context_id, user_message):
        url = f"{self.api_base_url}/context/completions/create"
        payload = {
            "context_id": context_id,
            "model": self.model,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
