import redis
import json
from typing import Dict, Any, Optional
import os
from utils.config_handler import redis_conf
from utils.logger_handler import logger

class RedisService:
    def __init__(self):
        # 内存存储作为Redis的备选方案
        self.memory_store = {}
        try:
            self.redis_client = redis.Redis(
                host=redis_conf.get("host", "localhost"),
                port=redis_conf.get("port", 6379),
                db=redis_conf.get("db", 0),
                password=redis_conf.get("password", None)
            )
            # 测试连接
            self.redis_client.ping()
            self.use_redis = True
        except Exception as e:
            logger.warning(f"Redis connection error: {e}")
            logger.info("Using memory store as fallback")
            self.redis_client = None
            self.use_redis = False
    
    def set_session(self, session_id: str, data: Dict[str, Any], expire: int = 3600) -> bool:
        """设置会话数据"""
        if self.use_redis:
            try:
                key = f"session:{session_id}"
                self.redis_client.setex(key, expire, json.dumps(data))
                return True
            except Exception as e:
                logger.warning(f"Redis set session error: {e}")
                # 失败时使用内存存储
                self.memory_store[session_id] = data
                return True
        else:
            # 使用内存存储
            self.memory_store[session_id] = data
            return True
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话数据"""
        if self.use_redis:
            try:
                key = f"session:{session_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
                # 如果Redis中没有，尝试从内存存储中获取
                return self.memory_store.get(session_id, {})
            except Exception as e:
                logger.warning(f"Redis get session error: {e}")
                # 失败时使用内存存储
                return self.memory_store.get(session_id, {})
        else:
            # 使用内存存储
            return self.memory_store.get(session_id, {})
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话数据"""
        if self.use_redis:
            try:
                key = f"session:{session_id}"
                self.redis_client.delete(key)
                # 同时从内存存储中删除
                if session_id in self.memory_store:
                    del self.memory_store[session_id]
                return True
            except Exception as e:
                logger.warning(f"Redis delete session error: {e}")
                # 失败时从内存存储中删除
                if session_id in self.memory_store:
                    del self.memory_store[session_id]
                return True
        else:
            # 使用内存存储
            if session_id in self.memory_store:
                del self.memory_store[session_id]
            return True
    
    def update_session(self, session_id: str, data: Dict[str, Any], expire: int = 3600) -> bool:
        """更新会话数据"""
        # 先获取现有会话数据
        existing_data = self.get_session(session_id) or {}
        # 更新数据
        existing_data.update(data)
        # 保存更新后的数据
        return self.set_session(session_id, existing_data, expire)
