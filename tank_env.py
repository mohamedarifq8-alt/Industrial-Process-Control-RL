"""محرك الخزان المطابق لبيئة التدريب الأصلية للوكيل."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_VALUES = (-0.5, 0.0, 0.5)
ACTION_LABELS = ("شفط", "إيقاف", "تعبئة")
TARGET_LEVEL = 5.0


class TankEnv(gym.Env[np.ndarray, int]):
    """خزان مستمر مع متغيري حالة: المستوى والسرعة.

    هذه المعادلات منسوخة من بيئة التدريب حتى تكون الحالة التي تراها الشبكة
    أثناء التشغيل مطابقة للحالة التي تعلمت منها الأوزان:

    - الفعل يسبب تسارعاً مقداره -0.5 أو 0 أو +0.5.
    - السرعة تتأثر باحتكاك 0.8.
    - المستوى يتحرك وفق السرعة.
    - يوجد تسريب احتمالي 20% بمقدار 0.2.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100, seed: int | None = None) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0], dtype=np.float32),
            high=np.array([10.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_level = TARGET_LEVEL
        self.velocity = 0.0
        self.step_count = 0
        self.last_action = 1
        self.last_reward = 0.0
        self.leak_active = False

    @property
    def level(self) -> float:
        """اسم عرض متوافق مع لوحة HMI."""
        return self.current_level

    @property
    def target_level(self) -> float:
        return TARGET_LEVEL

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.current_level = float(
            options.get("level", self.rng.uniform(2.0, 8.0))
        )
        self.velocity = float(options.get("velocity", 0.0))
        self.step_count = 0
        self.last_action = 1
        self.last_reward = 0.0
        self.leak_active = False
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(np.clip(action, 0, self.action_space.n - 1))
        self.last_action = action
        force = ACTION_VALUES[action]

        # نفس ترتيب العمليات في التدريب: دفع، احتكاك، ثم حركة المستوى.
        self.velocity += force
        self.velocity *= 0.8
        self.current_level += self.velocity

        self.leak_active = bool(self.rng.random() < 0.20)
        if self.leak_active:
            self.current_level -= 0.2

        self.current_level = float(np.clip(self.current_level, 0.0, 10.0))
        self.velocity = float(np.clip(self.velocity, -2.0, 2.0))
        self.step_count += 1

        terminated = bool(
            self.current_level <= 0.0 or self.current_level >= 10.0
        )
        truncated = self.step_count >= self.max_steps
        error = abs(self.current_level - TARGET_LEVEL)
        self.last_reward = -10.0 if terminated else -float(error)

        return (
            self._get_observation(),
            self.last_reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self) -> np.ndarray:
        return np.array([self.current_level, self.velocity], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "level": self.current_level,
            "velocity": self.velocity,
            "target_level": TARGET_LEVEL,
            "level_error": self.current_level - TARGET_LEVEL,
            "reward": self.last_reward,
            "leak_active": self.leak_active,
            "action_label": ACTION_LABELS[self.last_action],
            "step": self.step_count,
        }

    def close(self) -> None:
        """لا توجد موارد خارجية."""


def heuristic_action(observation: np.ndarray) -> int:
    """سياسة احتياطية بسيطة بنفس معنى أفعال الوكيل المدرب."""
    level, velocity = float(observation[0]), float(observation[1])
    if level > TARGET_LEVEL + 0.2 or velocity > 0.35:
        return 0
    if level < TARGET_LEVEL - 0.2 or velocity < -0.35:
        return 2
    return 1