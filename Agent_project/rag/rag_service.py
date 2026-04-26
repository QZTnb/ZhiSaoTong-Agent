
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.token_budget import TokenBudgetManager


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.token_budget = TokenBudgetManager()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        try:
            # 检查模型是否初始化成功
            if self.model is None:
                return "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"
            
            # 重置Token预算
            self.token_budget.reset_usage()
            
            # 检索文档
            context_docs = self.retriever_docs(query)
            
            # 优化检索结果，确保不超过Token预算
            optimized_docs = self.token_budget.optimize_rag_retrieval(query, [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in context_docs
            ])
            
            # 构建上下文
            context = ""
            counter = 0
            for doc in optimized_docs:
                counter += 1
                context += f"【参考资料{counter}】: 参考资料：{doc['page_content']} | 参考元数据：{doc['metadata']}\n"
            
            # 优化提示词
            optimized_input = self.token_budget.optimize_prompt(query)
            
            # 执行总结
            try:
                response = self.chain.invoke(
                    {
                        "input": optimized_input,
                        "context": context,
                    }
                )
            except Exception as model_error:
                print(f"Model invocation error: {str(model_error)}")
                return "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"
            
            # 更新Token使用情况
            self.token_budget.update_usage(query)
            self.token_budget.update_usage(context)
            self.token_budget.update_usage(response)
            
            return response
        except Exception as e:
            print(f"Error in rag_summarize: {str(e)}")
            import traceback
            traceback.print_exc()
            return "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
