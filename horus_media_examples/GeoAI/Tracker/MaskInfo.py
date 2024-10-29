from dataclasses import dataclass, field

@dataclass
class MaskInfo:
    
    id:int = 0
    mask: any = None
    mask_tracked: any = None
    id_hist:str = ""
    timestamp:int = 0.0
    area:int = 0
    bbox: list = field(default_factory=lambda: [])
    bbox_tracked: list = field(default_factory=lambda: [])

    history_mask: list = field(default_factory=lambda: [])
    history_area: list = field(default_factory=lambda: [])
    history_timestamp: list = field(default_factory=lambda: [])
    history_groundpoint: list = field(default_factory=lambda: [])
