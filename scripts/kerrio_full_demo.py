#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kerrio_full_demo.py - 完整的 Kerrio.AI 7阶段旅程演示脚本

这个脚本演示了 Mayo Clinic - Kerrio User Journey PDF 中定义的所有7个阶段。

使用方法:
1. 启动服务器: uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8080 --reload
2. 运行此脚本: python scripts/kerrio_full_demo.py
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080/api"
USER_ID = "mayo_demo_user"

def print_section(title):
    """打印美观的章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_subsection(title):
    """打印子章节标题"""
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}\n")

def chat(msg, show_full_reply=False):
    """发送聊天消息"""
    print(f"👤 User: {msg}")
    response = requests.post(f"{BASE_URL}/chat", json={
        "user_id": USER_ID,
        "user_msg": msg
    })
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return None
    
    data = response.json()
    reply = data.get('reply', '')
    
    if show_full_reply:
        print(f"🤖 Kerrio: {reply}\n")
    else:
        print(f"🤖 Kerrio: {reply[:150]}...\n")
    
    return data

def check_status():
    """检查旅程状态"""
    response = requests.get(f"{BASE_URL}/journey/{USER_ID}")
    if response.status_code != 200:
        print(f"❌ Error checking status: {response.text}")
        return {}
    
    data = response.json()
    print(f"📊 当前阶段: {data['current_stage']}")
    print(f"📊 可以前进: {data['can_advance']}")
    print(f"📊 对话回合数: {data['conversation_turns']}")
    return data

def advance_stage():
    """前进到下一阶段"""
    response = requests.post(f"{BASE_URL}/journey/advance", json={"user_id": USER_ID})
    if response.status_code != 200:
        print(f"❌ Error advancing: {response.text}")
        return False
    
    data = response.json()
    if data['success']:
        print(f"✅ 成功前进到: {data['new_stage']}")
        return True
    else:
        print(f"⚠️  无法前进: {data['message']}")
        return False

def main():
    """主演示流程"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║      Kerrio.AI - Mayo Clinic 7阶段临床旅程演示                    ║
║      Digital Cognitive Clinic (非AI聊天机器人)                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    time.sleep(1)
    
    # 重置用户数据（可选）
    try:
        requests.post(f"{BASE_URL}/reset", json={"user_id": USER_ID})
        print("🔄 已重置用户数据\n")
    except Exception as e:
        print(f"⚠️  无法连接服务器: {e}")
        print("请确保服务器正在运行: uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8080 --reload")
        sys.exit(1)
    
    time.sleep(1)
    
    # ==================== 阶段 1: Registration ====================
    print_section("阶段 1: REGISTRATION（注册验证）")
    print("📖 PDF要求: 客户作为受邀嘉宾注册，验证邀请码")
    print("📖 目的: 建立严肃性和排他性，过滤高意向客户\n")
    
    check_status()
    
    print("\n尝试在未验证的情况下前进...")
    advance_stage()
    
    print("\n✉️  验证邀请码: KERRIO-VIP")
    response = requests.post(f"{BASE_URL}/journey/validate", json={
        "user_id": USER_ID,
        "invite_code": "KERRIO-VIP"
    })
    result = response.json()
    print(f"验证结果: {result['success']} - {result['message']}")
    
    print("\n现在再次尝试前进...")
    if advance_stage():
        time.sleep(1)
    
    # ==================== 阶段 2: History Collection ====================
    print_section("阶段 2: HISTORY COLLECTION（三大支柱历史收集）")
    print("📖 PDF要求: 收集跨越三大支柱的综合个人历史")
    print("📖 这不是问卷调查 - 而是引导式的、叙事丰富的过程")
    print("📖 三大支柱:")
    print("   1. History: 人生事件、形成性经历、模式")
    print("   2. Psychology & Philosophy: 信念、价值观、意义结构、情感连线")
    print("   3. Physiology: 睡眠、压力、健康、能量、限制\n")
    
    check_status()
    
    print_subsection("Pillar 1: History（历史 - 人生事件与形成性经历）")
    chat("Hi Kerrio, I want to optimize my life and reach my full potential.")
    time.sleep(1)
    
    chat("When I was a child, my parents were both successful professionals. They had extremely high expectations for me. I always felt intense pressure to be perfect and never disappoint them.")
    time.sleep(1)
    
    print_subsection("Pillar 2: Psychology & Philosophy（心理与哲学）")
    chat("I believe that if I'm not achieving something significant every day, I'm wasting my potential. Success is extremely important to me. I value excellence and making a meaningful impact on the world.")
    time.sleep(1)
    
    chat("I've always thought that rest is for people who don't have ambition. I feel guilty when I'm not being productive.")
    time.sleep(1)
    
    print_subsection("Pillar 3: Physiology（生理状态）")
    chat("I typically sleep about 5 hours a night, sometimes less. I often feel stressed and overwhelmed. My energy levels are very inconsistent - sometimes I'm highly energized, but then I crash and feel exhausted.")
    time.sleep(1)
    
    # 查看收集的历史
    print_subsection("查看收集的历史数据")
    response = requests.get(f"{BASE_URL}/journey/history/{USER_ID}")
    if response.status_code == 200:
        history = response.json()
        print("📚 History Pillar:")
        print(f"  - 人生事件: {len(history['history_pillar']['life_events'])} 个")
        print(f"  - 形成性经历: {len(history['history_pillar']['formative_experiences'])} 个")
        
        print("\n🧠 Psychology & Philosophy Pillar:")
        print(f"  - 信念: {len(history['psychology_philosophy_pillar']['beliefs'])} 个")
        print(f"  - 价值观: {len(history['psychology_philosophy_pillar']['values'])} 个")
        
        print("\n💪 Physiology Pillar:")
        print(f"  - 睡眠质量: {history['physiology_pillar']['sleep_quality']}")
        print(f"  - 压力水平: {history['physiology_pillar']['stress_level']}")
    
    print("\n尝试前进到咨询阶段...")
    if advance_stage():
        time.sleep(1)
    
    # ==================== 阶段 3: Consultation ====================
    print_section("阶段 3: CONSULTATION（咨询 - 发现盲点）")
    print("📖 PDF要求: 通过结构化互动:")
    print("   - 澄清历史中的模糊之处")
    print("   - 发现客户可能看不到的盲点")
    print("   - 识别静态历史无法揭示的模式")
    print("📖 关键: 客户历史与临床笔记分开维护\n")
    
    check_status()
    
    print_subsection("展示抵抗（Resistance）")
    chat("I often start ambitious projects but rarely finish them. But I don't think that's really a problem - I just have very high standards and only want to complete things that are truly excellent.")
    time.sleep(1)
    
    print_subsection("展示回避/转移（Deflection - Blind Spot）")
    chat("Anyway, I don't really want to talk about that. Let's move on and discuss my future goals instead.")
    time.sleep(1)
    
    chat("Sometimes people tell me I'm too hard on myself, but that's not true. I just know what I'm capable of.")
    time.sleep(1)
    
    # 查看临床笔记
    print_subsection("查看临床笔记（Clinician Notes）")
    print("📖 PDF强调: 与客户历史分开维护")
    response = requests.get(f"{BASE_URL}/journey/notes/{USER_ID}")
    if response.status_code == 200:
        notes = response.json()
        print(f"\n🔍 临床观察数量: {len(notes['session_insights'])} 个")
        
        for i, insight in enumerate(notes['session_insights'][:5], 1):
            print(f"\n  洞察 {i}:")
            print(f"    类别: {insight['category']}")
            print(f"    观察: {insight['observation'][:100]}...")
        
        if notes['blind_spots_identified']:
            print(f"\n  识别的盲点: {len(notes['blind_spots_identified'])} 个")
    
    print("\n尝试前进到诊断阶段...")
    if advance_stage():
        time.sleep(1)
    
    # ==================== 阶段 4: Diagnosis ====================
    print_section("阶段 4: DIAGNOSIS（诊断 - 最重要的阶段）")
    print("📖 PDF要求: 这是最重要的阶段，必须:")
    print("   1. 识别核心限制和瓶颈")
    print("   2. 构建认知连线图（个性化大脑模型）")
    print("   3. 解释为什么问题存在，而不仅仅是问题看起来是什么")
    print("   4. 推荐教育视频")
    print("📖 关键: 理解是永久改变的前提条件\n")
    
    check_status()
    
    print_subsection("生成诊断")
    response = requests.get(f"{BASE_URL}/journey/diagnosis/{USER_ID}")
    if response.status_code == 200:
        diagnosis_data = response.json()
        diagnosis = diagnosis_data['diagnosis']
        
        print("\n🎯 核心限制 (Core Constraints):")
        for constraint in diagnosis['core_constraints']:
            print(f"  - {constraint}")
        
        print("\n🚧 瓶颈 (Bottlenecks):")
        for bottleneck in diagnosis['bottlenecks']:
            print(f"  - {bottleneck}")
        
        print("\n🔍 根本原因 (Root Causes):")
        for cause in diagnosis['root_causes']:
            print(f"  - {cause}")
        
        print(f"\n📝 诊断解释 (WHY, not just WHAT):")
        print(f"  {diagnosis['explanation']}")
        
        print(f"\n🎥 推荐的教育视频:")
        for video in diagnosis['recommended_videos']:
            print(f"  - [{video['video_id']}] {video['title']}")
            print(f"    相关性: {video['relevance']}\n")
    
    print_subsection("查看认知连线图 (Cognitive Wiring Map)")
    print("📖 PDF: 每个客户独特的个性化大脑模型")
    response = requests.get(f"{BASE_URL}/journey/map/{USER_ID}")
    if response.status_code == 200:
        cog_map = response.json()
        print(f"\n🧠 认知连线图统计:")
        print(f"  - 节点数量: {len(cog_map.get('nodes', []))}")
        print(f"  - 边数量: {len(cog_map.get('edges', []))}")
        print(f"  - 摘要: {cog_map.get('summary', 'N/A')[:100]}...")
        print(f"  - 最后更新: {cog_map.get('last_updated', 'N/A')}")
    
    print("\n✅ 确认理解诊断")
    print("📖 PDF: 客户必须理解诊断才能继续")
    response = requests.post(f"{BASE_URL}/journey/diagnosis/confirm/{USER_ID}")
    if response.status_code == 200:
        print(f"确认结果: {response.json()['success']}")
    
    print("\n尝试前进到治疗方案阶段...")
    if advance_stage():
        time.sleep(1)
    
    # ==================== 阶段 5: Treatment Proposal ====================
    print_section("阶段 5: TREATMENT PROPOSAL（治疗方案）")
    print("📖 PDF要求: 只有在客户理解诊断后才提出治疗")
    print("📖 计划必须:")
    print("   - 个性化到认知连线图")
    print("   - 结构化、有序列、深思熟虑")
    print("   - 包括认知重新连线图（专利申请中）")
    print("📖 没有通用的'教练建议'\n")
    
    check_status()
    
    print_subsection("获取治疗计划")
    response = requests.get(f"{BASE_URL}/journey/treatment/{USER_ID}")
    if response.status_code == 200:
        treatment_data = response.json()
        treatment = treatment_data['treatment_proposal']
        
        print(f"\n⏱️  预计持续时间: {treatment['estimated_duration_weeks']} 周")
        print(f"💊 干预措施数量: {len(treatment['interventions'])}")
        
        if treatment['interventions']:
            print(f"\n干预措施详情:")
            for i, intervention in enumerate(treatment['interventions'][:3], 1):
                print(f"  {i}. {intervention['name']}")
                print(f"     描述: {intervention['description'][:80]}...")
                print(f"     频率: {intervention['frequency']}")
        
        if treatment.get('rewiring_map'):
            print_subsection("认知重新连线图 (Patent Pending Cognitive Rewiring Map)")
            print("📖 PDF: 专利申请中的技术 - Kerrio的核心差异化")
            
            rewiring = treatment['rewiring_map']
            print(f"\n🔄 当前连线模式:")
            print(f"  {rewiring['current_wiring']}")
            
            print(f"\n🎯 目标连线模式:")
            print(f"  {rewiring['target_wiring']}")
            
            print(f"\n📊 重新连线步骤: {len(rewiring['rewiring_steps'])} 个")
            for i, step in enumerate(rewiring['rewiring_steps'][:3], 1):
                print(f"  步骤 {i}: {step['name']}")
                print(f"    {step['description'][:80]}...")
            
            print(f"\n进度: {rewiring['progress'] * 100:.1f}%")
    
    print("\n✅ 接受治疗方案")
    print("📖 PDF: 客户必须接受治疗才能继续")
    response = requests.post(f"{BASE_URL}/journey/proposal/accept", json={"user_id": USER_ID})
    if response.status_code == 200:
        print(f"接受结果: {response.json()['success']}")
    
    print("\n尝试前进到治疗阶段...")
    if advance_stage():
        time.sleep(1)
    
    # ==================== 阶段 6: Treatment ====================
    print_section("阶段 6: TREATMENT（治疗进行中）")
    print("📖 PDF要求: 使用基于神经科学的干预:")
    print("   - 认知重新连线练习")
    print("   - 结构化的认知和行为重新校准")
    print("   - 针对性的心理重新连线")
    print("📖 目标: 结构性改变，而非动机顺从\n")
    
    status = check_status()
    
    print_subsection("模拟治疗进展")
    # 这里可以模拟完成重新连线步骤
    chat("I've been practicing the awareness exercise you suggested. I'm starting to notice when I fall into perfectionist thinking patterns.")
    time.sleep(1)
    
    # 如果需要达到50%进度才能前进到监测阶段，这里可以更新进度
    # (在实际应用中，可能需要完成多个步骤)
    
    # ==================== 阶段 7: Monitoring ====================
    print_section("阶段 7: MONITORING（监测与进度评估）")
    print("📖 PDF要求: 纵向监测进度，对比:")
    print("   - 原始认知连线图")
    print("   - 原始限制")
    print("   - 性能和幸福指标")
    print("📖 闭环: 诊断 → 治疗 → 重新评估\n")
    
    print_subsection("提交监测反馈")
    response = requests.post(f"{BASE_URL}/journey/monitoring/submit", json={
        "user_id": USER_ID,
        "metrics": {
            "sleep_hours": 6.5,
            "stress_level": "moderate",
            "energy": "improving",
            "pattern_recognition": "aware",
            "perfectionism_episodes": "decreasing"
        },
        "notes": "Client shows increased awareness of perfectionism patterns. Sleep has slightly improved. Still working on self-compassion."
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 监测反馈已提交")
        if result.get('rediagnose_needed'):
            print("⚠️  系统检测到需要重新诊断（闭环机制）")
    
    # ==================== 完成总结 ====================
    print_section("旅程完成 - 最终状态")
    
    final_status = check_status()
    
    print(f"\n📊 统计数据:")
    print(f"  - 最终阶段: {final_status['current_stage']}")
    print(f"  - 总对话回合: {final_status['conversation_turns']}")
    print(f"  - 临床见解数: {final_status['clinician_insights_count']}")
    print(f"  - 人生事件: {final_status['client_history_summary']['life_events_count']}")
    print(f"  - 信念: {final_status['client_history_summary']['beliefs_count']}")
    print(f"  - 价值观: {final_status['client_history_summary']['values_count']}")
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║  ✅ Mayo Clinic 7阶段临床旅程演示完成！                           ║
║                                                                   ║
║  项目已完整实现 PDF 中定义的所有要求:                            ║
║  ✓ 诊断优先（而非动机/参与）                                     ║
║  ✓ 理解是永久改变的前提                                          ║
║  ✓ 客户历史与临床笔记分离                                        ║
║  ✓ 认知连线图 (个性化大脑模型)                                   ║
║  ✓ 专利申请中的认知重新连线图                                    ║
║  ✓ 结构性改变（非外在动机）                                      ║
║  ✓ 闭环：诊断 → 治疗 → 重新评估                                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n💡 提示:")
    print("  - 可以访问 http://localhost:8080/ 查看 Web 界面")
    print("  - 用户数据保存在: runs/kerrio_profiles/")
    print("  - 对话日志保存在: runs/chat_logs/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
