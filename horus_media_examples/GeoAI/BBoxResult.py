import string
from dataclasses import dataclass

@dataclass
class BBoxResult:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.0
    classID: int = 1
    className: string = ""
    index: int = 0
    uuid: string = ""