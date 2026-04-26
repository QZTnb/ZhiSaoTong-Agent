from typing import Dict, List, Optional
import tiktoken
from model.factory import chat_model
from utils.config_handler import rag_conf

class TokenBudgetManager:
    def __init__(self, max_budget: int = 4096):
        self.max_budget = max_budget
        self.current_usage = 0
        # 使用配置文件中的模型名称，默认为gpt-3.5-turbo
        model_name = rag_conf.get("chat_model_name", "gpt-3.5-turbo")
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            # 如果模型名称不支持，使用默认的tokenizer
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def count_tokens(self, text: str) -> int:
        """计算文本的Token数量"""
        return len(self.tokenizer.encode(text))
    
    def update_usage(self, text: str) -> None:
        """更新Token使用情况"""
        self.current_usage += self.count_tokens(text)
    
    def reset_usage(self) -> None:
        """重置Token使用情况"""
        self.current_usage = 0
    
    def get_remaining_budget(self) -> int:
        """获取剩余的Token预算"""
        return max(0, self.max_budget - self.current_usage)
    
    def is_within_budget(self, text: str) -> bool:
        """检查文本是否在预算范围内"""
        return self.current_usage + self.count_tokens(text) <= self.max_budget
    
    def optimize_rag_retrieval(self, query: str, documents: List[Dict], max_tokens: Optional[int] = None) -> List[Dict]:
        """优化RAG检索结果，确保不超过Token预算"""
        if max_tokens is None:
            max_tokens = self.get_remaining_budget() // 2  # 预留一半预算给模型响应
        
        optimized_docs = []
        current_tokens = 0
        
        # 按相关性排序（假设documents已经按相关性排序）
        for doc in documents:
            doc_tokens = self.count_tokens(doc.get("page_content", ""))
            if current_tokens + doc_tokens <= max_tokens:
                optimized_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break
        
        return optimized_docs
    
    def optimize_prompt(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """优化提示词，确保不超过Token预算"""
        if max_tokens is None:
            max_tokens = self.get_remaining_budget() // 2
        
        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens <= max_tokens:
            return prompt
        
        # 截断提示词，保留开头和结尾的重要信息
        tokens = self.tokenizer.encode(prompt)
        truncated_tokens = tokens[:max_tokens-10]  # 预留一些空间
        return self.tokenizer.decode(truncated_tokens) + "..."
