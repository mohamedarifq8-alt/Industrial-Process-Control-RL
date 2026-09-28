import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# ==========================================
# المرحلة الأولى: بناء ذاكرة الوكيل (Experience Replay)
# ==========================================
class ReplayMemory:
    def __init__(self, capacity):

        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        return random.sample(self.memory, batch_size)

    def __len__(self):

        return len(self.memory)        
        
  
# ==========================================
# المرحلة الثانية: بناء العقل (الشبكة العصبية DQN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        # تفعيل الوراثة من مكتبة PyTorch لجلب كل العمليات الرياضية المعقدة
        super(DQN, self).__init__()
        
        # الطبقة الأولى: تستقبل قراءة حساس المستوى (مدخل) وتمررها لـ 64 عقدة معالجة
        self.fc1 = nn.Linear(state_dim, 64)
        
        # الطبقة الثانية (الطبقة المخفية): لزيادة عمق التفكير وفهم العلاقات غير الخطية (مثل التسريب)
        self.fc2 = nn.Linear(64, 64)
        
        # الطبقة الثالثة (المخرجات): تُخرج القيم النهائية للأفعال (شفط، إيقاف، تعبئة)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        # مسار تدفق الإشارة الكهربائية (البيانات)
        # نستخدم دالة التنشيط ReLU (مثل الدايود) لتمرير الإشارات الإيجابية فقط وكسر النمط الخطي
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # المخرج النهائي لا يحتاج "دايود" لأننا نريد القيم الخام (Q-Values) حتى لو كانت سالبة
        return self.fc3(x)
        

import torch.optim as optim

# ==========================================
# المرحلة الثالثة: بناء المتحكم المركزي (DQNAgent)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 1. إعداد معاملات التحكم (Hyperparameters)
        self.gamma = 0.95           # النظرة المستقبلية (نفس القيمة التي استخدمتها في جدول Q)
        self.epsilon = 1.0          # نبدأ بالاستكشاف العشوائي الكامل للمصنع
        self.epsilon_min = 0.01     # الحد الأدنى للاستكشاف
        self.epsilon_decay = 0.995  # معدل تقليل العشوائية مع مرور الوقت
        self.batch_size = 32        # حجم العينة المسحوبة من الذاكرة في كل تدريب
        self.lr = 0.001             # سرعة تعلم الشبكة العصبية
        
        # 2. إعداد العتاد (المعالج)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3. بناء الشبكات العصبية (Main Controller & Target Controller)
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        # نسخ الأوزان الأولية للشبكة الهدف وجعلها في وضع التقييم (ثابتة)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 4. تجهيز الذاكرة والمُحسّن الرياضي (Adam Optimizer)
        self.memory = ReplayMemory(2000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # التعديل: تحويل الحالة المزدوجة [المستوى, السرعة] مباشرة إلى تنسور
        # نستخدم unsqueeze(0) لإضافة بُعد الدفعة (Batch) لتصبح (1, 2)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # التعديل الجوهري: إزالة التطبيع (القسمة على 10) وإزالة unsqueeze(1) للحالات
        # لأن states أصبحت مصفوفة (Batch, 2) بشكل تلقائي
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)

        # الأفعال والمكافآت لا تزال قيماً مفردة، لذا نحتفظ بـ unsqueeze(1) لها
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # حساب القيم الحالية من الشبكة الرئيسية
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        # حساب القيم الهدف باستخدام الشبكة الثابتة
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))

        # حساب نسبة الخطأ (Loss)
        loss = F.mse_loss(current_q_values, target_q_values)

        # التحديث العكسي
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # تقليل العشوائية
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # نقل الخبرة من الشبكة الرئيسية إلى الشبكة الهدف (استقرار النظام)
        self.target_net.load_state_dict(self.policy_net.state_dict())


from gymnasium import spaces
import numpy as np
import sys

class ContinuousTankEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # الأفعال متقطعة كما هي: 0 (شفط)، 1 (إيقاف)، 2 (تعبئة)
        self.action_space = spaces.Discrete(3)
        
        # التعديل الأول: الحالات مستمرة وتحتوي على متغيرين (المستوى، السرعة)
        # المستوى من 0 إلى 10، والسرعة من -2 إلى 2
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0]), 
            high=np.array([10.0, 2.0]), 
            dtype=np.float32
        )
        self.current_level = 5.0
        self.velocity = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # نبدأ بمستوى عشوائي دقيق (مثلاً 3.45)
        self.current_level = self.np_random.uniform(2.0, 8.0)
        self.velocity = 0.0
        return np.array([self.current_level, self.velocity], dtype=np.float32), {}

    def step(self, action):
        # التعديل الثاني: الفعل يولد قوة دفع (Acceleration) وليس تغييراً مباشراً للمستوى
        force = 0.0
        if action == 0:
            force = -0.5 # قوة شفط
        elif action == 2:
            force = 0.5  # قوة تعبئة

        # تحديث السرعة بناءً على الدفع، مع إضافة "احتكاك" لكي لا تتسارع للأبد (0.8)
        self.velocity += force
        self.velocity *= 0.8 

        # تحديث المستوى بناءً على السرعة (القصور الذاتي)
        self.current_level += self.velocity

        # فيزياء التسريب (تسريب دقيق)
        if self.np_random.random() < 0.20:
            self.current_level -= 0.2

        # القيود الفيزيائية
        self.current_level = max(0.0, min(10.0, self.current_level))
        self.velocity = max(-2.0, min(2.0, self.velocity))

        terminated = bool(self.current_level <= 0.0 or self.current_level >= 10.0)

        # التعديل الثالث: المكافأة المستمرة
        error = abs(self.current_level - 5.0)
        if terminated:
            reward = -10.0
        else:
            reward = -error # كلما اقترب من 5.0 بدقة، قل العقاب

        return np.array([self.current_level, self.velocity], dtype=np.float32), reward, terminated, False, {}

    def render(self):
        # شاشة HMI معدلة لتعرض الأرقام بدقة الفاصلة العشرية
        sys.stdout.write(f"\rالمستوى: {self.current_level:.2f} | السرعة: {self.velocity:.2f} | [")
        for i in range(11):
            if i == 5:
                sys.stdout.write("🎯" if abs(self.current_level - 5.0) < 0.2 else "|")
            elif i <= int(self.current_level):
                sys.stdout.write("🟦")
            else:
                sys.stdout.write("  ")
        sys.stdout.write("]  ")
        sys.stdout.flush()


# --- إعداد التدريب ---
env = ContinuousTankEnv()
agent = DQNAgent(state_dim=2, action_dim=3) # لاحظ تغيير state_dim إلى 2

EPISODES = 300       # عدد الورديات التدريبية
TARGET_UPDATE = 10   # تحديث الشبكة الهدف كل 10 ورديات
MAX_STEPS = 100      # طول الوردية الواحدة

print("بدء تدريب العقل العميق (DQN) على إدارة المصنع...")

for e in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < MAX_STEPS:
        # 1. اتخاذ القرار
        action = agent.act(state)
        
        # 2. تنفيذ الأمر في المصنع
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 3. تخزين التجربة في الذاكرة (Experience Replay)
        agent.memory.push(state, action, reward, next_state, done)
        
        # 4. التدريب (التفكير) بعد كل خطوة
        agent.learn()
        
        state = next_state
        total_reward += reward
        step_count += 1
        
    # 5. تحديث الشبكة الهدف بشكل دوري لضمان استقرار التعلم
    if e % TARGET_UPDATE == 0:
        agent.update_target_network()
        
    if (e + 1) % 20 == 0:
        print(f"الوردية: {e+1}/{EPISODES} | المكافأة الإجمالية: {total_reward:.2f} | نسبة العشوائية (الاستكشاف): {agent.epsilon:.2f}")

print("\nتم تدريب المتحكم بنجاح!")


from IPython import display
import time

# ==========================================
# مرحلة الاختبار الحي (Live Testing - HMI)
# ==========================================
state, _ = env.reset()
done = False
step_count = 0

# إيقاف العشوائية نهائياً للاختبار
agent.epsilon = 0.0

while not done and step_count < 50:
    # 1. مسح الشاشة السابقة (هذا ما يعطي تأثير التحديث اللحظي)
    display.clear_output(wait=True)
    
    # 2. اتخاذ القرار وتنفيذه
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    step_count += 1
    action_names = ["شفط 🔻", "إيقاف ⏸️", "تعبئة 🔺"]
    
    # 3. طباعة لوحة المراقبة
    print("--- تشغيل المصنع بالنظام العميق (DQN) ---")
    print(f"الدورة: {step_count}/50")
    print(f"قرار العقل: {action_names[action]}")
    
    # استدعاء دالة الرسم (التي تستخدم \r للبقاء في نفس السطر)
    env.render()
    
    state = next_state
    done = terminated or truncated
    
    # تأخير زمني بسيط لنستطيع رؤية التحديث بالعين المجردة
    time.sleep(0.4)

print("\n\n======================")
if not done:
    print("نجاح باهر! 🏆 المتحكم العميق حافظ على استقرار الخزان.")
else:
    print("النظام فشل 🚨.")