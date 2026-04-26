from langgraph.graph import StateGraph, END
from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 定义状态类型
class AgentState(Dict[str, Any]):
    messages: List[Dict[str, str]]
    query: str
    agent_type: str
    response: str
    session_id: Optional[str]
    react_turns: int
    max_react_turns: int

# 主Agent
class MainAgent:
    def __init__(self):
        # 初始化内存存储
        self.memory_store = {}
        # 初始化RAG服务
        self.rag_service = RagSummarizeService()
        # 初始化查询缓存
        self.query_cache = {}
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("route", self.route_query)
        workflow.add_node("pre_sales_agent", self.pre_sales_agent)
        workflow.add_node("after_sales_agent", self.after_sales_agent)
        workflow.add_node("operation_agent", self.operation_agent)
        workflow.add_node("direct_answer", self.direct_answer)  # 直接回答模式
        workflow.add_node("react_think", self.react_think)
        workflow.add_node("react_act", self.react_act)
        workflow.add_node("react_observe", self.react_observe)
        workflow.add_node("finalize", self.finalize_response)
        
        # 添加边
        workflow.set_entry_point("route")
        workflow.add_conditional_edges(
            "route",
            lambda x: x["agent_type"],
            {
                "pre_sales": "pre_sales_agent",
                "after_sales": "after_sales_agent",
                "operation": "operation_agent",
                "direct": "direct_answer"  # 直接回答模式
            }
        )
        workflow.add_edge("pre_sales_agent", "react_think")
        workflow.add_edge("after_sales_agent", "react_think")
        workflow.add_edge("operation_agent", "react_think")
        workflow.add_edge("react_think", "react_act")
        workflow.add_edge("react_act", "react_observe")
        workflow.add_conditional_edges(
            "react_observe",
            lambda x: x["react_turns"] < x["max_react_turns"],
            {
                True: "react_think",
                False: "finalize"
            }
        )
        workflow.add_edge("direct_answer", "finalize")
        workflow.add_edge("finalize", END)
        
        # 编译图
        self.graph = workflow.compile()
    
    def route_query(self, state: AgentState) -> AgentState:
        """根据查询内容路由到不同的子Agent"""
        query = state["query"].lower()
        
        # 检查是否命中缓存
        cache_key = query
        if cache_key in self.query_cache:
            return {**state, "agent_type": "direct", "react_turns": 0, "max_react_turns": 0}
        
        # 简单问题直接回答
        if any(keyword in query for keyword in ["你好", "你是谁", "自我介绍", "问候"]):
            return {**state, "agent_type": "direct", "react_turns": 0, "max_react_turns": 0}
        
        # 简单的路由逻辑
        if any(keyword in query for keyword in ["买", "购买", "价格", "推荐", "型号", "选择"]):
            return {**state, "agent_type": "pre_sales", "react_turns": 0, "max_react_turns": 1}  # 减少ReAct轮数
        elif any(keyword in query for keyword in ["问题", "故障", "维修", "退换", "售后"]):
            return {**state, "agent_type": "after_sales", "react_turns": 0, "max_react_turns": 1}  # 减少ReAct轮数
        elif any(keyword in query for keyword in ["维护", "保养", "清洁", "使用", "操作"]):
            return {**state, "agent_type": "operation", "react_turns": 0, "max_react_turns": 1}  # 减少ReAct轮数
        else:
            # 默认直接回答模式
            return {**state, "agent_type": "direct", "react_turns": 0, "max_react_turns": 0}
    
    def get_system_prompt(self, agent_type: str) -> str:
        """根据Agent类型获取系统提示词"""
        prompts = {
            "pre_sales": "你是智扫通智能机器人客服的售前顾问，负责解答用户关于扫地机器人的选购问题。",
            "after_sales": "你是智扫通智能机器人客服的售后顾问，负责解答用户关于扫地机器人的故障和售后问题。",
            "operation": "你是智扫通智能机器人客服的运维顾问，负责解答用户关于扫地机器人的维护和使用问题。"
        }
        return prompts.get(agent_type, prompts["pre_sales"])
    
    def pre_sales_agent(self, state: AgentState) -> AgentState:
        """售前Agent"""
        system_prompt = self.get_system_prompt("pre_sales")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]
        return {**state, "messages": messages}
    
    def after_sales_agent(self, state: AgentState) -> AgentState:
        """售后Agent"""
        system_prompt = self.get_system_prompt("after_sales")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]
        return {**state, "messages": messages}
    
    def operation_agent(self, state: AgentState) -> AgentState:
        """运维Agent"""
        system_prompt = self.get_system_prompt("operation")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]
        return {**state, "messages": messages}
    
    def direct_answer(self, state: AgentState) -> AgentState:
        """直接回答模式，避免ReAct循环"""
        query = state["query"]
        cache_key = query.lower()
        
        # 检查缓存
        if cache_key in self.query_cache:
            return {**state, "response": self.query_cache[cache_key]}
        
        # 处理简单问题
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["你好", "你是谁", "自我介绍", "问候"]):
            response = "我是智扫通智能机器人客服，专注于解答扫地机器人的售前、售后和运维问题。"
            self.query_cache[cache_key] = response
            return {**state, "response": response}
        
        # 调用RAG服务
        try:
            response = self.rag_service.rag_summarize(query)
        except Exception as e:
            # 出错时使用默认回答
            response = "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"
        
        # 更新缓存
        self.query_cache[cache_key] = response
        
        return {**state, "response": response}
    
    def react_think(self, state: AgentState) -> AgentState:
        """ReAct思考步骤"""
        # 构建思考提示词
        think_prompt = """
        你是一个使用ReAct模式的智能助手。请按照以下步骤处理用户的请求：
        1. 分析用户的问题和当前的上下文
        2. 思考如何解决这个问题
        3. 决定是否需要调用工具获取更多信息
        4. 如果需要调用工具，请明确说明要调用的工具和参数
        5. 如果不需要调用工具，请直接给出答案
        """
        
        # 添加思考提示词到消息
        messages = state["messages"]
        messages.append({"role": "system", "content": think_prompt})
        
        # 生成思考内容
        response = chat_model.invoke(messages)
        messages.append({"role": "assistant", "content": response.content})
        
        return {**state, "messages": messages, "react_turns": state["react_turns"] + 1}
    
    def react_act(self, state: AgentState) -> AgentState:
        """ReAct行动步骤"""
        # 分析思考内容，决定是否调用工具
        last_message = state["messages"][-1]["content"]
        
        # 检查是否需要调用RAG工具
        if "rag" in last_message.lower() or "检索" in last_message.lower() or "参考资料" in last_message.lower():
            # 调用RAG工具
            query = state["query"]
            rag_response = self.rag_service.rag_summarize(query)
            
            # 添加工具执行结果到消息
            messages = state["messages"]
            messages.append({"role": "tool", "content": f"RAG检索结果: {rag_response}"})
            
            return {**state, "messages": messages}
        else:
            # 不需要调用工具，直接进入观察步骤
            return state
    
    def react_observe(self, state: AgentState) -> AgentState:
        """ReAct观察步骤"""
        # 分析工具执行结果，生成观察内容
        messages = state["messages"]
        
        # 生成观察内容
        observe_prompt = """
        请基于之前的对话和工具执行结果，总结当前的情况，并决定是否需要进一步的行动：
        1. 分析工具执行结果
        2. 评估是否已经获得足够的信息
        3. 决定是否需要进一步调用工具
        4. 如果信息足够，给出最终答案
        """
        
        messages.append({"role": "system", "content": observe_prompt})
        response = chat_model.invoke(messages)
        messages.append({"role": "assistant", "content": response.content})
        
        return {**state, "messages": messages}
    
    def finalize_response(self, state: AgentState) -> AgentState:
        """最终处理响应"""
        # 从消息中提取最终响应
        messages = state["messages"]
        
        # 检查是否已有响应
        if "response" in state:
            return state
        
        # 检查messages是否为空
        if not messages:
            # 如果messages为空，使用默认响应
            return {**state, "response": "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"}
        
        last_message = messages[-1]["content"]
        
        # 如果最后一条消息是观察结果，直接作为响应
        if "观察" in last_message or "总结" in last_message:
            response = last_message
        else:
            # 否则，使用模型生成最终响应
            try:
                response = chat_model.invoke(messages).content
            except Exception as e:
                # 模型调用失败时使用默认响应
                response = "我是智扫通智能机器人客服，专注于解答扫地机器人的相关问题。"
        
        return {**state, "response": response}
    
    def execute(self, query: str, session_id: str = None) -> str:
        """执行查询"""
        try:
            # 从内存存储获取会话历史（如果有）
            session_data = self.memory_store.get(session_id, {}) if session_id else {}
            
            # 构建初始状态
            initial_state = {
                "query": query,
                "session_id": session_id,
                "messages": []
            }
            
            # 执行查询
            result = self.graph.invoke(initial_state)
            
            # 更新会话状态
            if session_id:
                session_data["last_query"] = query
                session_data["last_response"] = result["response"]
                session_data["agent_type"] = result["agent_type"]
                self.memory_store[session_id] = session_data
            
            return result["response"]
        except Exception as e:
            print(f"Error in execute: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"抱歉，处理您的请求时出错：{str(e)}"
    
    def execute_stream(self, query: str, session_id: str = None):
        """流式执行查询"""
        import time
        start_time = time.time()
        max_execution_time = 60  # 60秒超时
        
        try:
            # 从内存存储获取会话历史（如果有）
            session_data = self.memory_store.get(session_id, {}) if session_id else {}
            
            # 构建初始状态
            initial_state = {
                "query": query,
                "session_id": session_id,
                "messages": []
            }
            
            # 执行查询，添加超时检查
            result = None
            def execute_with_timeout():
                nonlocal result
                result = self.graph.invoke(initial_state)
            
            # 使用线程执行，避免阻塞
            import threading
            thread = threading.Thread(target=execute_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(max_execution_time)
            
            if not result:
                # 超时，返回友好提示
                yield "抱歉，处理超时，请稍后重试。"
                return
            
            full_response = result["response"]
            
            # 模拟流式输出，每次输出一个字符
            for char in full_response:
                yield char
                time.sleep(0.005)  # 减少延迟，提高响应速度
            
            # 更新会话状态
            if session_id:
                session_data["last_query"] = query
                session_data["last_response"] = full_response
                session_data["agent_type"] = result["agent_type"]
                self.memory_store[session_id] = session_data
        except Exception as e:
            print(f"Error in execute_stream: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"抱歉，处理您的请求时出错：{str(e)}"
