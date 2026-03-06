from dataclasses import dataclass, field
from typing import Optional, List

from .base_elements import Artifact
from ..registry import register

@register('audio')
@dataclass(kw_only=True)
class Audio(Artifact):
    path: str
    type: str = 'audio'
