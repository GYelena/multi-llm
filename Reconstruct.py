import json
import re
import time
import concurrent.futures
from openai import OpenAI

# ==========================================
# 1. 物理连接与注册表配置
# ==========================================
client = OpenAI(api_key="vllm-token", base_url="http://localhost:8000/v1")

# 预定义模型注册表 (完全对齐 Paper 中的 3 个专属小模型)
MODEL_REGISTRY = {
    "agent_a_explorer": {
        "description": "启发探索者：擅长数据收集、提出先验假设与宏观思路拆解。",
        "vllm_id": "my-local-model", 
        "local_path": "/root/autodl-tmp/model_agent_a" 
    },
    "agent_b_reasoner": {
        "description": "严密推演者：擅长逻辑推演、数学计算、复杂代码生成与硬核执行。",
        "vllm_id": "my-local-model",
        "local_path": "/root/autodl-tmp/model_agent_b"
    },
    "agent_c_critic": {
        "description": "严苛审查者：擅长逻辑挑错、寻找数据冲突与边界条件验证。",
        "vllm_id": "my-local-model",
        "local_path": "/root/autodl-tmp/model_agent_c"
    }
}

# ==========================================
# 2. 核心执行引擎类
# ==========================================
class DynamicDAGExecutor:
    def __init__(self, original_query):
        self.original_query = original_query
        self.results = {}      
        self.completed_ids = set()
        self.pending_tasks = []      # 🌟 核心：动态任务队列
        self.execution_history = []  # 🌟 核心：全局上下文，用于评估高熵
        self.log_file = "workflow_log.md"
        self.global_start_time = 0

    def write_log(self, step_name, model_info, input_content, raw_output):
        """记录模型间的详细沟通细节，用于论文的 Case Study 分析"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n# {step_name}\n")
            f.write(f"**所选模型/角色**: {model_info}\n")
            f.write(f"### [输入上下文内容]\n{input_content}\n\n")
            f.write(f"### [模型输出]\n{raw_output}\n")
            f.write("\n" + "="*60 + "\n")

    def strip_think(self, text):
        """剥离可能存在的思维链标签"""
        if not text: return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def robust_json_extract(self, text):
        """鲁棒的 JSON 提取器"""
        content = text.split("</think>")[-1].strip() if "</think>" in text else text
        try:
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1).replace('\n', '').replace('\\', ''))
        except: return None
        return None

    def ask_llm(self, model_id, system_role, user_content, temperature=0.6):
        """统一的大模型调用接口"""
        physical_id = MODEL_REGISTRY.get(model_id, {}).get("vllm_id", "my-local-model")
        # 中心模型 (Planner) 使用基础大模型，不走小模型专精路由
        if model_id == "central_planner": 
            physical_id = "my-local-model"
            
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

    # ------------------------------------------
    # Step 1: 显式的初始静态图谱规划
    # ------------------------------------------
    def initial_planning(self):
        model_menu = "\n".join([f"- {k}: {v['description']}" for k, v in MODEL_REGISTRY.items()])
        planner_prompt = f"""你是一个高级中心规划专家。请将用户问题拆解为有逻辑依赖的子任务。
        专家模型列表：\n{model_menu}
        请输出 JSON 格式：
        {{
          "tasks": [
            {{"id": "T1", "target_model": "agent_a_explorer", "query": "任务指令", "depends_on": []}}
          ]
        }}"""
        
        print("\n" + "="*50)
        print(">> [1/4] 中心模型正在进行初始 DAG 规划...")
        raw_output = self.ask_llm("central_planner", planner_prompt, f"请拆解任务: {self.original_query}", temperature=0.1)
        plan = self.robust_json_extract(raw_output)
        
        if plan and "tasks" in plan:
            self.pending_tasks = plan["tasks"]
            self.write_log("初始任务规划", "Central Planner", self.original_query, raw_output)
            print(f"   [OK] 规划成功，构建了 {len(self.pending_tasks)} 个初始节点。具体如下：")
            for t in self.pending_tasks:
                t_id = t.get('id', 'N/A')
                target = t.get('target_model', 'N/A')
                query = t.get('query', '')[:40] + "..."
                depends = t.get('depends_on', [])
                dep_str = f"(依赖: {depends})" if depends else "(无依赖)"
                print(f"      - 节点 [{t_id}] 分配给 <{target}> | 指令: {query} {dep_str}")
        else:
            print("   [FAILED] 规划解析失败，请检查模型输出格式。")

    # ------------------------------------------
    # Step 2 & 3: 状态嗅探与动态重构 (MDP 核心机制)
    # ------------------------------------------
    def assess_and_reconstruct(self, latest_task_id, target_model, latest_result):
        """Step 2 & 3: 显式的 MDP 状态评估与动态重构 (严格逐层评估版)"""
        print("\n" + "-"*50)
        print(f">> 中心模型正在深度评估节点 [{latest_task_id}] (<{target_model}>) 的输出...")
        
        context_str = "\n".join(self.execution_history[-3:]) # 取最近的历史作为决策上下文
        eval_prompt = """你是一个具备 MDP 动态路由能力的中心调度模型。
        请阅读当前的执行上下文。
        1. 如果发现逻辑死胡同、数据严重冲突或无法解决的数学报错，输出 RECONSTRUCT 并生成替代方案。
        2. 如果当前执行顺利，逻辑自洽，没有致命矛盾，输出 CONTINUE。
        
        请输出 JSON 格式：
        {
            "action": "RECONSTRUCT", // 或 "CONTINUE"
            "thought": "你的决策依据，请明确说明为什么决定重构，或者为什么认为当前状态健康可以继续...",
            "new_tasks": [ // 如果是 RECONSTRUCT，提供修正任务；若是 CONTINUE，此项留空 []
                {"id": "T4_new", "target_model": "agent_b_reasoner", "query": "新的替代指令", "depends_on": []}
            ]
        }"""

        # 唤醒中心模型进行深度思考
        raw_output = self.ask_llm("central_planner", eval_prompt, f"当前执行上下文:\n{context_str}", temperature=0.1)
        decision = self.robust_json_extract(raw_output)
        self.write_log("MDP 状态评估与重构", "Central Planner", context_str, raw_output)

        if decision:
            action = decision.get("action", "CONTINUE")
            thought = decision.get("thought", "未提供理由")
            
            if action == "RECONSTRUCT":
                print(f"   [决策: RECONSTRUCT (动态重构触发)]")
                print(f"      依据: {thought}")
                
                # 清空旧队列，挂载新分支
                self.pending_tasks.clear() 
                new_tasks = decision.get("new_tasks", [])
                self.pending_tasks.extend(new_tasks)
                
                print(f"      已挂起失效路径。插入了 {len(new_tasks)} 个新节点，新图谱如下：")
                for t in new_tasks:
                    t_id = t.get('id', 'N/A')
                    target = t.get('target_model', 'N/A')
                    query = t.get('query', '')[:40] + "..."
                    print(f"         - 节点 [{t_id}] 分配给 <{target}> | 指令: {query}")
                print("-" * 50)
                return True
            else:
                print(f"   [决策: CONTINUE (维持原计划)]")
                print(f"      依据: {thought}")
                print("-" * 50)
                return False
        else:
            print("   [评估失败] 无法解析 JSON，默认继续。")
            print("-" * 50)
            return False

    # ------------------------------------------
    # 主循环流转 (反应式执行引擎)
    # ------------------------------------------
    def run(self):
        self.global_start_time = time.time()
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# 自动编排执行全记录\n\n")

        self.initial_planning()
        if not self.pending_tasks: return "执行终止。"

        print(f"\n>> [2/4] 反应式执行引擎启动 (Reactive Execution Engine)...")
        
        while self.pending_tasks:
            # 找到依赖已满足的就绪任务
            ready_tasks = [t for t in self.pending_tasks if all(d in self.completed_ids for d in t.get('depends_on', []))]
            
            if not ready_tasks: 
                print("   [ERROR] 检测到任务依赖死锁！当前待办任务依赖的前置节点尚未完成或被剔除。")
                break

            current_task = ready_tasks[0]
            self.pending_tasks.remove(current_task) # 弹出队列
            
            t_id = current_task.get('id')
            target_m = current_task.get('target_model', 'agent_a_explorer')
            
            context = "".join([f"\n--- 步骤{d}的结果 ---\n{self.results[d]}\n" for d in current_task.get('depends_on', [])])
            full_query = f"上下文参考: {context}\n具体工作指令: {current_task.get('query')}"
            
            print(f"      [执行] 节点 {t_id} -> {target_m} 正在处理...")
            raw_res = self.ask_llm(target_m, f"你是专业的{target_m}", full_query)
            clean_res = self.strip_think(raw_res)
            
            self.write_log(f"执行步骤 {t_id}", target_m, full_query, raw_res)
            self.results[t_id] = clean_res 
            self.completed_ids.add(t_id)
            
            # 记录到全局上下文，供中心模型“嗅探”
            self.execution_history.append(f"[{t_id} | {target_m}] 输出: {clean_res[:200]}...")
            print(f"      [✓] 节点 {t_id} 完成")

            # 触发中心模型的 MDP 状态评估拦截器 (带上 target_m 参数)
            self.assess_and_reconstruct(t_id, target_m, clean_res)

        print(f"\n>> [3/4] DAG 遍历完毕，开始汇总...")
        history_lean = "\n\n".join([f"## 节点{k}:\n{v}" for k, v in self.results.items()])
        final_report = self.ask_llm("central_planner", "你负责做最终结论总结，整合各专家的推演结果。", f"全过程推演数据:\n{history_lean}")
        self.write_log("最终汇总报告", "Central Planner", "全过程数据摘要", final_report)
        
        print(f"   [✓] 任务结束！总耗时: {round(time.time() - self.global_start_time, 2)}s")
        return final_report

# ==========================================
# 启动测试入口
# ==========================================
if __name__ == "__main__":
    # 使用一个高复杂度数学推理任务，测试小模型之间真实的逻辑论辩
    q = "推导某非高斯厚尾分布下贝叶斯后验的闭式解析解，并严谨地验证其概率公理与收敛性。"
    
    executor = DynamicDAGExecutor(q)
    res = executor.run()
    
    print("\n" + "="*20 + " 最终结论预览 " + "="*20)
    print(executor.strip_think(res))