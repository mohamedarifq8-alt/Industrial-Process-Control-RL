import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TankControlEnv(gym.Env):
    """
    بيئة تحكم في مستوى خزان صناعي.
    الهدف: الحفاظ على مستوى السائل عند 5.
    الحالات: من 0 إلى 10.
    الأفعال: 0 (تفريغ)، 1 (إيقاف)، 2 (تعبئة).
    """
    
    def __init__(self):
        super().__init__() # استدعاء دالة التهيئة للمكتبة الأم
        
        # 1. تعريف الأفعال (المشغلات / Actuators)
        # لدينا 3 أفعال متقطعة (0, 1, 2)
        self.action_space = spaces.Discrete(3)
        
        # 2. تعريف الحالات (الحساسات / Sensors)
        # لدينا 11 حالة متقطعة (من 0 إلى 10)
        self.observation_space = spaces.Discrete(11)
        
        # متغير داخلي لحفظ مستوى السائل الحالي
        self.current_level = 5 
        
    def reset(self, seed=None, options=None):
        # تهيئة المولد العشوائي الخاص بالبيئة (مهم لتثبيت النتائج أثناء التجارب)
        super().reset(seed=seed)
        
        # بدء "وردية العمل" بمستوى عشوائي للماء لتدريب الوكيل على كافة الظروف
        # نختار رقماً من 1 إلى 9 (تجنبنا 0 و 10 لكي لا تنتهي المحاولة قبل أن تبدأ)
        self.current_level = self.np_random.integers(1, 10)
        
        # معلومات إضافية (اختيارية) يمكن إرجاعها للمراقبة
        info = {"message": "تم إعادة ضبط الخزان"}
        
        # حسب معايير Gymnasium الحديثة، دالة reset يجب أن تُرجع (state, info)
        return self.current_level, info

print("تم بناء هيكل البيئة (TankControlEnv) بنجاح مع دالتي التهيئة وإعادة الضبط!")



import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

class TankControlEnv(gym.Env):
    """بيئة تحكم في مستوى خزان صناعي"""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)      # 0: شفط, 1: إيقاف, 2: تعبئة
        self.observation_space = spaces.Discrete(11) # المستويات من 0 إلى 10
        self.current_level = 5 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_level = self.np_random.integers(1, 10)
        return self.current_level, {}

    def step(self, action):
        # 1. تنفيذ أوامر التحكم (Actuation)
        if action == 0:
            self.current_level -= 1  # تشغيل مضخة الشفط
        elif action == 2:
            self.current_level += 1  # تشغيل مضخة التعبئة
        # إذا كان الفعل 1 (إيقاف)، لن تتغير القيمة من جهة الوكيل

        # 2. إضافة الفيزياء العشوائية (التسريب/التبخر - Disturbance)
        # توليد رقم عشوائي بين 0.0 و 1.0. إذا كان أقل من 0.2 فهذا يعني احتمال 20%
        if self.np_random.random() < 0.20:
            self.current_level -= 1

        # 3. تطبيق القيود الفيزيائية (Saturation/Clipping)
        # نضمن أن المستوى لا ينزل تحت 0 ولا يتجاوز 10 مهما حدث
        self.current_level = max(0, min(10, self.current_level))

        # 4. التحقق من شروط الفشل (Termination)
        terminated = bool(self.current_level == 0 or self.current_level == 10)
        truncated = False # نستخدمها لاحقاً إذا أردنا وضع حد أقصى للزمن

        # 5. حساب المكافأة المتدرجة (Shaped Reward)
        if terminated:
            reward = -1.0 # فشل كارثي للنظام
        elif self.current_level == 5:
            reward = 1.0  # وصول مثالي للهدف
        else:
            # عقاب متدرج: كلما كبر الخطأ، زاد العقاب السالب
            error = abs(self.current_level - 5)
            reward = -0.1 * error

        return self.current_level, reward, terminated, truncated, {}

    def render(self):
        # رسم الخزان نصياً للمراقبة المرئية (HMI)
        sys.stdout.write(f"\rالمستوى الحالي: [")
        for i in range(11):
            if i == 5:
                sys.stdout.write("🎯" if self.current_level == 5 else "|") # علامة المنتصف
            elif i <= self.current_level:
                sys.stdout.write("🟦") # ماء
            else:
                sys.stdout.write("  ") # فراغ
        sys.stdout.write(f"] ({self.current_level}/10)")
        sys.stdout.flush()

print("تم بناء البيئة بالكامل بنجاح! جميع الدوال (init, reset, step, render) جاهزة.")


import numpy as np
import random

# 1. تهيئة المصنع وجدول Q
env = TankControlEnv()
state_size = env.observation_space.n  # 11 حالة
action_size = env.action_space.n      # 3 أفعال
q_table = np.zeros((state_size, action_size))

# 2. معاملات التدريب (Hyperparameters)
num_episodes = 10000
max_steps_per_episode = 100 # نعتبر أن الوردية الواحدة مدتها 100 ثانية
learning_rate = 0.1
discount_rate = 0.95 # رفعنا النظرة المستقبلية قليلاً لكي يهتم بالاستقرار على المدى الطويل

# معاملات الاستكشاف
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

def choose_action(state, current_epsilon):
    if random.uniform(0, 1) > current_epsilon:
        return np.argmax(q_table[state, :]) # استغلال
    else:
        return env.action_space.sample()    # استكشاف

print("جاري تدريب متحكم الذكاء الاصطناعي... يرجى الانتظار...")

# 3. حلقة التدريب (Training Loop)
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    for step in range(max_steps_per_episode):
        # اتخاذ القرار
        action = choose_action(state, epsilon)
        
        # التفاعل مع المصنع (لاحظ أن البيئة هي التي تحسب المكافأة الآن!)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # تحديث العقل (معادلة بيلمان)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action]
        )
        
        state = new_state
        if done:
            break
            
    # تقليل العشوائية
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("تم التدريب بنجاح!")
print("\n--- جدول Q النهائي للمتحكم ---")
# طباعة الجدول بأسماء الأفعال لتسهيل القراءة
import pandas as pd
df = pd.DataFrame(q_table, columns=["شفط (0)", "إيقاف (1)", "تعبئة (2)"])
df.index.name = "مستوى السائل"
print(np.round(df, 3))


import time
from IPython import display
import numpy as np

# تهيئة المصنع للاختبار
state, _ = env.reset()
done = False
step_count = 0

print("بدء تشغيل المصنع بنظام تحكم الذكاء الاصطناعي...")
time.sleep(2)

while not done and step_count < 50: # نختبره لمدة 50 دورة
    # العرض المرئي
    display.clear_output(wait=True)
    
    # اتخاذ القرار الحتمي من جدول Q
    action = np.argmax(q_table[state, :])
    
    # تنفيذ الأمر في المصنع
    new_state, reward, terminated, truncated, _ = env.step(action)
    
    # تحديد إذا ما كان هناك تسريب قد حدث (لمراقبة سرعة استجابة الوكيل)
    # إذا أمرنا بالإيقاف (1) ولكن المستوى نزل، أو أمرنا بالتعبئة (2) وبقي المستوى ثابتاً.. فهذا تسريب
    leak_occurred = False
    if (action == 1 and new_state < state) or (action == 2 and new_state == state) or (action == 0 and new_state < state -1):
        leak_occurred = True

    step_count += 1
    
    # واجهة المراقبة (HMI)
    action_names = ["شفط 🔻", "إيقاف ⏸️", "تعبئة 🔺"]
    print(f"--- وردية العمل: الدورة {step_count}/50 ---")
    print(f"قرار المتحكم: {action_names[action]}")
    if leak_occurred:
        print("⚠️ تنبيه: حدث تسريب مفاجئ للسائل في هذه الدورة!")
    else:
        print("الوضع مستقر.")
        
    env.render()
    
    state = new_state
    done = terminated or truncated
    
    time.sleep(0.5)

print("\n\n======================")
if not done:
    print("نجاح باهر! 🏆 المتحكم حافظ على استقرار الخزان طوال فترة الاختبار دون أي فشل.")
else:
    print("النظام فشل 🚨 (تأكد من عدم تغيير كود البيئة).")