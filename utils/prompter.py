"""
Code from alpaca-lora
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose # 是否打印调试信息
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"

        # 拼接模板文件路径（模板文件存放在"templates"目录下，格式为JSON）
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")

        # 加载模板JSON文件
        with open(file_name) as fp:
            self.template = json.load(fp)

        # 打印模板描述（如果开启verbose）
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            # 有上下文输入时，使用prompt_input模板，替换{instruction}和{input}
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            # 无上下文输入时，使用prompt_no_input模板，替换{instruction}
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        # 如果提供了label（如训练时的参考答案），则追加到提示词后
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # 以response_split为分隔符，提取模型输出的响应部分
        return output.split(self.template["response_split"])[1].strip()
