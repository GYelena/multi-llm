import json
import re
import time
import concurrent.futures
from openai import OpenAI

# 1. 配置物理连接
client = OpenAI(api_key="vllm-token", base_url="http://localhost:8000/v1")

# 2. 预定义模型注册表 
# 模拟了三个模型，目前由于只部署了一个模型，所以 vllm_id 都指向同一个
MODEL_REGISTRY = {
    "researcher_model": {
        "description": "擅长多源信息搜集、事实核查与基础背景分析。",
        "vllm_id": "my-local-model", # 物理调用 ID
        "local_path": "/root/autodl-tmp/model_researcher_v1" # 物理存储路径（备查）
    },
    "analyst_model": {
        "description": "擅长逻辑推演、因果关系分析以及对复杂数据的定性定量评估。",
        "vllm_id": "my-local-model",
        "local_path": "/root/autodl-tmp/model_analyst_v1"
    },
    "writer_model": {
        "description": "擅长语言润色、结构化总结、文案优化以及行业建议的撰写。",
        "vllm_id": "my-local-model",
        "local_path": "/root/autodl-tmp/model_writer_v1"
    }
}

class LocalDAGExecutor:
    def __init__(self, original_query):
        self.original_query = original_query
        self.results = {}      
        self.completed_ids = set()
        self.log_file = "workflow_log.md"
        self.global_start_time = 0

    def write_log(self, step_name, model_info, input_content, raw_output):
        """记录模型间的详细沟通细节"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n# {step_name}\n")
            f.write(f"**所选模型 ID**: {model_info}\n")
            f.write(f"### [输入上下文内容]\n{input_content}\n\n")
            f.write(f"### [模型原始完整输出 (含思考过程)]\n\n{raw_output}\n")
            f.write("\n" + "="*60 + "\n")

    def strip_think(self, text):
        if not text: return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def ask_llm(self, model_id, system_role, user_content, temperature=0.6):
        # 根据逻辑 ID 获取物理 vLLM 模型名称
        physical_id = MODEL_REGISTRY.get(model_id, {}).get("vllm_id", "my-local-model")
        try:
            response = client.chat.completions.create(
                model=physical_id,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {e}"

    def robust_json_extract(self, text):
        content = text.split("</think>")[-1].strip() if "</think>" in text else text
        try:
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1).replace('\\n', '').replace('\\', ''))
        except: return None
        return None

    def plan_tasks(self):
        # 构造模型选择的上下文
        model_menu = "\n".join([f"- {k}: {v['description']}" for k, v in MODEL_REGISTRY.items()])
        
        planner_prompt = f"""你是一个任务规划专家。请将用户问题拆解为 2-4 个有逻辑依赖的子任务。
        
        ### 你可以调用的专家模型列表：
        {model_menu}

        ### 输出要求：
        必须输出 JSON 块，并为每个任务指派上述列表中的 target_model。
        {{
          "tasks": [
            {{"id": 1, "target_model": "researcher_model", "query": "任务内容", "depends_on": []}}
          ]
        }}
        """
        print("\n>> [1/3] 中心模型正在分析需求并匹配专家模型...")
        # 规划阶段使用默认模型
        raw_output = self.ask_llm("researcher_model", planner_prompt, f"请拆解: {self.original_query}", temperature=0.1)
        
        self.write_log("任务规划阶段", "Planner-Center", f"原始问题: {self.original_query}", raw_output)
        
        plan = self.robust_json_extract(raw_output)
        if not plan or "tasks" not in plan: return []
        
        tasks = plan["tasks"]
        print(f"   [OK] 规划成功，共 {len(tasks)} 个步骤。")
        for t in tasks:
            m_id = t.get('target_model', 'unknown')
            dep_str = f"(等待 {t.get('depends_on')})" if t.get('depends_on') else "(无依赖)"
            print(f"        - 步骤 {t.get('id')}: [{m_id}] {t.get('query')[:40]}... {dep_str}")
        
        return tasks

    def run(self):
        self.global_start_time = time.time()
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# 自动编排执行全记录\n\n")

        tasks_list = self.plan_tasks()
        if not tasks_list: return "规划解析失败。"

        print(f"\n>> [2/3] 正在按逻辑顺序分发任务给选定模型...")
        
        while len(self.completed_ids) < len(tasks_list):
            ready_tasks = [t for t in tasks_list if t.get('id') not in self.completed_ids and all(d in self.completed_ids for d in t.get('depends_on', []))]
            if not ready_tasks: break

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_task = {}
                for t in ready_tasks:
                    t_id = t.get('id')
                    target_m = t.get('target_model', 'researcher_model')
                    
                    context = "".join([f"\n--- 步骤{d}的结果 ---\n{self.results[d]}\n" for d in t.get('depends_on', [])])
                    full_query = f"上下文参考: {context}\n具体工作指令: {t.get('query')}"
                    
                    # 状态反馈
                    print(f"      [分发] 正在调用 {target_m} 处理步骤 {t_id}...")
                    
                    f = executor.submit(self.ask_llm, target_m, f"你是{target_m}", full_query)
                    future_to_task[f] = (t_id, target_m, full_query)

                for future in concurrent.futures.as_completed(future_to_task):
                    t_id, m_id, t_input = future_to_task[future]
                    raw_res = future.result()
                    
                    self.write_log(f"执行步骤 {t_id}", m_id, t_input, raw_res)
                    self.results[t_id] = raw_res 
                    self.completed_ids.add(t_id)
                    print(f"      [✓] 步骤 {t_id} 完成")

        print(f"\n>> [3/3] 汇总所有专家意见并生成报告...")
        history_lean = "\n\n".join([f"## 专家{k}结论:\n{self.strip_think(v)}" for k, v in self.results.items()])
        final_input = f"研究全数据汇总:\n{history_lean}"
        
        # 汇总阶段
        final_report = self.ask_llm("writer_model", "你是一个首席执行官，负责汇总报告。", final_input)
        self.write_log("最终汇总报告", "CEO (writer_model)", final_input, final_report)
        
        print(f"   [✓] 任务结束！总耗时: {round(time.time() - self.global_start_time, 2)}s")
        return final_report

if __name__ == "__main__":
    q = "调研2025年人形机器人的技术瓶颈，分析其对制造业就业的潜在影响，并给出一份行业建议。"
    executor = LocalDAGExecutor(q)
    res = executor.run()
    print("\n" + "="*20 + " 最终报告预览 " + "="*20)
    print(executor.strip_think(res))