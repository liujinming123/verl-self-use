# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import json
from typing import Any, Dict, List, Optional

from verl.tools.base_tool import BaseTool, ToolResponse
ToolExecutionResponse = ToolResponse


class SearchSimilarCasesTool(BaseTool):
    """Tool for searching similar historical cases."""

    def __init__(self, config: dict, tool_schema=None):
        """Initialize the tool.

        Args:
            config: Configuration dictionary containing history_data_path.
            tool_schema: Tool schema (optional).
        """
        self.history_data_path = config.get("history_data_path")
        self.history_data = self._load_history_data()
        self.tool_schema = tool_schema

    def _load_history_data(self) -> List[Dict[str, Any]]:
        """Load history data from CSV file.

        Returns:
            List of history data entries.
        """
        history_data = []
        try:
            with open(self.history_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    history_data.append({
                        "query": row.get("query", ""),
                        "label": int(row.get("label", 0)),
                        "think_label": row.get("think_label", "")
                    })
        except Exception as e:
            print(f"Error loading history data: {e}")
        return history_data

    async def create(self, create_kwargs: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
        """Create tool instance.

        Args:
            create_kwargs: Additional creation arguments.

        Returns:
            Instance ID and metadata.
        """
        instance_id = "search_similar_cases_instance"
        metadata = {"history_data_path": self.history_data_path}
        return instance_id, metadata

    async def execute(
        self, instance_id: str, tool_args: Any, agent_data: Optional[Any] = None
    ) -> tuple[ToolExecutionResponse, float, Dict[str, Any]]:
        """Execute the tool.

        Args:
            instance_id: Instance ID.
            tool_args: Tool arguments.
            agent_data: Agent data.

        Returns:
            Tool execution response, reward, and additional data.
        """
        keywords = tool_args.get("keywords", "")
        top_k = tool_args.get("top_k", 5)

        # Simple keyword matching for demonstration
        results = []
        for entry in self.history_data:
            if keywords.lower() in entry["query"].lower():
                results.append({
                    "对话": entry["query"],
                    "结果": "是" if entry["label"] == 1 else "否",
                    "原因": entry["think_label"]
                })
            if len(results) >= top_k:
                break

        # If no results found, return empty list
        if not results:
            results = [{"对话": "无匹配案例", "是否负反馈": "", "原因": ""}]

        response_text = json.dumps(results, ensure_ascii=False, indent=2)
        return ToolExecutionResponse(text=response_text), 0.0, {}

    async def release(self, instance_id: str):
        """Release tool instance.

        Args:
            instance_id: Instance ID.
        """
        pass

    @property
    def name(self) -> str:
        """Tool name."""
        return "search_similar_cases"
