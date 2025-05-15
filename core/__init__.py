from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class BaseComponent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
        self.initialize()

    @abstractmethod
    def initialize(self):
        """初始化组件"""
        pass

    @abstractmethod
    def process(self, inputs: Any) -> Any:
        """处理输入并返回输出"""
        pass

    def validate_config(self):
        """验证配置的有效性"""
        required_keys = getattr(self, 'required_config_keys', [])
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def update_config(self, new_config: Dict[str, Any]):
        """更新组件配置"""
        self.config.update(new_config)
        self.validate_config()    