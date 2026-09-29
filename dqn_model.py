"""شبكة DQN صغيرة وقابلة لإعادة الاستخدام للتحكم في خزان كيميائي."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch import nn


DEFAULT_STATE_SIZE = 2
DEFAULT_ACTION_SIZE = 3


class DQN(nn.Module):
    """شبكة Q قابلة للتشغيل على CPU أو GPU.

    هذه البنية مطابقة تماماً للشبكة المستخدمة في التدريب:
    fc1(2 → 64) ثم fc2(64 → 64) ثم fc3(64 → 3).
    """

    def __init__(
        self,
        state_size: int = DEFAULT_STATE_SIZE,
        action_size: int = DEFAULT_ACTION_SIZE,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        # أسماء الطبقات مهمة لأن ملف التدريب يحفظ المفاتيح:
        # fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias.
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """إرجاع قيم Q لكل فعل."""
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))
        return self.fc3(state)

    @torch.no_grad()
    def q_values(self, state: Any, device: torch.device | str = "cpu") -> torch.Tensor:
        """حساب قيم Q مع قبول مصفوفة NumPy أو Tensor."""
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return self(tensor)

    @torch.no_grad()
    def act(
        self,
        state: Any,
        device: torch.device | str = "cpu",
        epsilon: float = 0.0,
    ) -> int:
        """اختيار الفعل الأفضل أو فعل عشوائي بنسبة epsilon."""
        if epsilon > 0.0 and torch.rand(()) < epsilon:
            return int(torch.randint(self.action_size, (1,)).item())
        values = self.q_values(state, device=device)
        return int(torch.argmax(values, dim=1).item())

    def load_checkpoint(
        self,
        checkpoint: str | Path | BinaryIO,
        device: torch.device | str = "cpu",
    ) -> dict[str, list[str]]:
        """تحميل الأوزان من state_dict أو checkpoint يحتوي على state_dict.

        يعيد أسماء المفاتيح الناقصة وغير المتوقعة حتى تعرض الواجهة حالة التحميل
        بوضوح بدلاً من اعتبار ملف غير متوافق صالحاً بصمت.
        """
        try:
            payload = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            # توافق مع إصدارات PyTorch القديمة التي لا تدعم weights_only.
            payload = torch.load(checkpoint, map_location=device)

        state_dict = self._extract_state_dict(payload)
        normalized = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "policy_net.", "target_net."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            normalized[key] = value
        result = self.load_state_dict(normalized, strict=False)
        self.to(device)
        self.eval()
        return {
            "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys),
        }

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, nn.Module):
            return dict(payload.state_dict())
        if not isinstance(payload, dict):
            raise ValueError("ملف الأوزان لا يحتوي على state_dict صالح.")

        for key in (
            "state_dict",
            "model_state_dict",
            "q_network",
            "policy_net",
            "policy_net_state_dict",
            "target_net_state_dict",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate

        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
        raise ValueError(
            "لم يتم العثور على state_dict. استخدم state_dict أو model_state_dict."
        )


def build_dqn(
    weights_path: str | Path | BinaryIO | None = None,
    device: torch.device | str = "cpu",
) -> tuple[DQN, dict[str, list[str]] | None]:
    """إنشاء النموذج وتحميل الأوزان اختيارياً."""
    model = DQN()
    model.to(device)
    model.eval()
    if weights_path is None:
        return model, None
    report = model.load_checkpoint(weights_path, device=device)
    return model, report