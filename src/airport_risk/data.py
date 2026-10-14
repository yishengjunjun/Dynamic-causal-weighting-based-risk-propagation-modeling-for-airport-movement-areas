"""风险负载数据契约、论文节点目录与事件级实验划分。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

# 表1的编号与名称；两个“设备可靠性差”条目保留为不同节点。
HUMAN = [
    "Human factors",
    "Ground handling personnel and vehicle operator errors",
    "ATC personnel errors",
    "Flight crew factors",
    "Failure of ground personnel to detect hazards in time",
    "Unauthorized runway entry by ground personnel",
    "Incorrect runway entry",
    "Delayed response by ground personnel/drivers",
    "Inadequate professional competence of ground personnel/drivers",
    "Low airport familiarity among ground personnel/drivers",
    "Limited work experience of ground personnel/drivers",
    "Years of service among ground personnel/drivers",
    "Inadequate training for ground personnel/drivers",
    "Failure to detect/correct crew readback errors",
    "Failure to detect/respond to crew/vehicle anomalies",
    "ATC Non-compliance with procedures",
    "Delayed ATC instructions",
    "Substandard ATC radio communication",
    "Deficient ATC professional competence",
    "Limited ATC work experience",
    "Prolonged ATC reaction time",
    "Inadequate ATC personnel training",
    "Low years of experience among ATC personnel",
    "High workload for ATC personnel",
    "Insufficient number of ATC personnel",
    "Incorrect readback",
    "Flight crew unfamiliar with the airport",
    "Flight crew entering runway without clearance",
    "Delayed response by flight crew",
    "Flight crew failing to detect errors in time",
    "Poor air-ground communication skills of flight crew",
    "Prolonged crew reaction time",
    "Deficient crew professional competence",
    "Low years of service among flight crew",
    "Low pilot experience",
    "Inadequate flight crew training",
]
ENVIRONMENT = [
    "Environmental factors",
    "Slippery surface",
    "Low visibility",
    "High traffic volume",
    "Complex runway-taxiway layout",
    "Nighttime operations",
    "Precipitation",
    "Dust/sandstorm",
]
EQUIPMENT = [
    "Equipment factors",
    "Surveillance equipment failure",
    "Communication equipment failure",
    "Poor equipment reliability",
    "Ground lighting interruption",
    "Critical lighting failure",
    "Obstructed ground markings",
    "Electromagnetic interference",
    "Faded/illogical markings",
    "Poor equipment reliability",
    "Delayed maintenance",
    "Automated equipment failure",
    "Inoperative warning device",
]
OVERSIGHT = [
    "Oversight factors",
    "Low inspection frequency",
    "Low compliance rate with regulations",
    "Insufficient monitoring measures",
    "Insufficient oversight personnel",
    "Inadequate regulatory investment",
]


@dataclass
class RiskData:
    history: np.ndarray  # [事件, 历史步数, 节点]，风险负载值
    target: np.ndarray  # [事件, 节点]，下一时刻负载观测
    adjacency: np.ndarray  # [源节点, 目标节点]，二元邻接矩阵，对角线为零
    capacities: np.ndarray  # [节点]，严格为正
    impedance: np.ndarray  # [节点]，布尔阻抗标记
    grades: np.ndarray  # [节点]，由输入数据给定的灰关联等级
    years: np.ndarray  # [事件]
    airports: np.ndarray  # [事件]，保留数据源中的机场标识
    event_ids: np.ndarray  # [事件]
    node_ids: np.ndarray  # [节点]
    factor_values: np.ndarray  # [事件, 候选因子]，历史代理指标
    factor_reference: np.ndarray  # [事件]，历史参考值，不使用下一时刻目标
    metadata: dict
    sample_ids: np.ndarray | None = None  # [样本]，同一事件的多个窗口使用不同编号
    future_loads: np.ndarray | None = None  # [样本, 未来步数, 节点]，仅用于训练或评估

    def validate(self):
        h, y = self.history, self.target
        if h.ndim != 3 or h.shape[0] == 0 or h.shape[1] < 2 or y.shape != (h.shape[0], h.shape[2]):
            raise ValueError("history must be [events,H>=2,N]; target [events,N]")
        b, _, n = h.shape
        if n < 2 or self.adjacency.shape != (n, n):
            raise ValueError("adjacency shape mismatch or fewer than two nodes")
        if not np.isin(self.adjacency, [0, 1]).all() or np.diag(self.adjacency).any():
            raise ValueError("adjacency must be binary with no self-edges")
        if (
            self.capacities.shape != (n,)
            or self.impedance.shape != (n,)
            or self.grades.shape != (n,)
        ):
            raise ValueError("node attribute shape mismatch")
        if not np.isin(self.impedance, [0, 1]).all():
            raise ValueError("impedance must be a binary node-aligned vector")
        if (
            self.factor_values.ndim != 2
            or self.factor_values.shape[0] != b
            or self.factor_values.shape[1] == 0
        ):
            raise ValueError("factor_values must be [events,nonempty candidate factors]")
        if self.factor_reference.shape != (b,):
            raise ValueError("factor/reference row mismatch")
        for x in (h, y, self.capacities, self.grades, self.factor_values, self.factor_reference):
            if not np.isfinite(x).all():
                raise ValueError("data contains NaN/Inf; impute explicitly before importing")
        if np.any(h < 0) or np.any(y < 0) or np.any(self.capacities <= 0):
            raise ValueError("loads must be nonnegative; capacities strictly positive")
        if np.any(self.grades < 0) or self.grades.sum() <= 0:
            raise ValueError("grades must be nonnegative and sum to >0")
        for x in (self.years, self.airports, self.event_ids):
            if x.shape != (b,):
                raise ValueError("event metadata shape mismatch")
        if not np.isfinite(self.years).all() or np.any(self.years != np.floor(self.years)):
            raise ValueError("years must be finite integer calendar years")
        if self.sample_ids is None:
            if len(np.unique(self.event_ids)) != b:
                raise ValueError("duplicate event IDs require explicit unique sample_ids")
            self.sample_ids = np.asarray(self.event_ids).astype("U").copy()
        if self.sample_ids.shape != (b,) or len(np.unique(self.sample_ids)) != b:
            raise ValueError("sample_ids must be a unique sample-aligned vector")
        for event in np.unique(self.event_ids):
            rows = self.event_ids == event
            if len(np.unique(self.years[rows])) != 1 or len(np.unique(self.airports[rows])) != 1:
                raise ValueError("all windows of an event must share its year and airport")
        if self.future_loads is not None:
            future = self.future_loads
            if (
                future.ndim != 3
                or future.shape[0] != b
                or future.shape[1] < 1
                or future.shape[2] != n
                or not np.isfinite(future).all()
                or np.any(future < 0)
            ):
                raise ValueError("future_loads must be finite nonnegative [samples,F>=1,N]")
            if not np.array_equal(future[:, 0], y):
                raise ValueError("target must equal the first future_loads step")
        if self.node_ids.shape != (n,) or len(np.unique(self.node_ids)) != n:
            raise ValueError("node IDs must be unique")
        for identifier_name in ("event_ids", "sample_ids", "node_ids", "airports"):
            identifiers = np.asarray(getattr(self, identifier_name))
            if np.any(np.char.str_len(identifiers.astype(str)) == 0):
                raise ValueError("event, node and airport identifiers cannot be empty")
            # 从 Pandas 转换的编号数组常为对象类型；统一存为 Unicode，
            # 确保加载 NPZ 时无需启用 pickle。
            setattr(self, identifier_name, identifiers.astype("U"))
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self.metadata.get("provenance") not in ("synthetic", "authorized_real"):
            raise ValueError("metadata.provenance must be synthetic or authorized_real")
        mapping = self.metadata.get("node_factor_indices")
        if mapping is None and self.metadata["provenance"] == "authorized_real":
            raise ValueError("authorized_real data requires explicit node_factor_indices")
        if mapping is not None:
            ix = np.asarray(mapping)
            if (
                ix.shape != (n,)
                or not np.isfinite(ix).all()
                or not np.equal(ix, np.floor(ix)).all()
                or np.any(ix < 0)
                or np.any(ix >= self.factor_values.shape[1])
                or len(np.unique(ix)) != n
            ):
                raise ValueError(
                    "node_factor_indices must be N distinct valid candidate column indices"
                )
        try:
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON-serializable data") from error
        return self

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.__dataclass_fields__:
            if name == "metadata":
                continue
            value = getattr(self, name)
            if value is None:
                digest.update(f"{name}:None".encode("utf-8"))
                continue
            x = np.asarray(value)
            if x.dtype.kind in ("O", "U", "S"):
                x = x.astype("U")
            digest.update(name.encode("utf-8"))
            digest.update(str((x.shape, x.dtype.str)).encode("ascii"))
            digest.update(np.ascontiguousarray(x).tobytes())
        digest.update(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False, ensure_ascii=False).encode(
                "utf-8"
            )
        )
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        arrays = {k: v for k, v in self.__dict__.items() if k != "metadata" and v is not None}
        np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            fields = {k: z[k] for k in cls.__dataclass_fields__ if k != "metadata" and k in z}
            fields["metadata"] = json.loads(str(z["metadata_json"]))
        return cls(**fields).validate()


def node_catalog():
    names = HUMAN + ENVIRONMENT + EQUIPMENT + OVERSIGHT
    ids = [
        f"{p}{i}"
        for p, items in [("a", HUMAN), ("b", ENVIRONMENT), ("c", EQUIPMENT), ("d", OVERSIGHT)]
        for i in range(1, len(items) + 1)
    ]
    return np.array(ids), names


def split_indices(
    data, *, protocol="stratified", seed=0, train_fraction=0.7, test_year=None, holdout=None
):
    """返回事件级切分索引；此处不拟合任何参数。"""
    data.validate()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data.target))
    if protocol == "stratified":
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be in (0,1)")
        train = []
        test = []
        for year in np.unique(data.years):
            events = np.unique(data.event_ids[data.years == year])
            rng.shuffle(events)
            if len(events) < 2:
                raise ValueError("each year needs >=2 events for the stratified split")
            cut = min(len(events) - 1, max(1, int(np.floor(len(events) * train_fraction))))
            train.extend(indices[np.isin(data.event_ids, events[:cut])])
            test.extend(indices[np.isin(data.event_ids, events[cut:])])
        return np.sort(train), np.array([], dtype=int), np.sort(test)
    if protocol == "rolling":
        if test_year is None:
            raise ValueError("rolling split needs test_year")
        if not np.isfinite(test_year) or test_year != int(test_year):
            raise ValueError("test_year must be a finite integer year")
        tr = indices[(data.years >= test_year - 12) & (data.years <= test_year - 4)]
        va = indices[(data.years >= test_year - 3) & (data.years < test_year)]
        te = indices[data.years == test_year]
    elif protocol == "airport":
        if not isinstance(holdout, str) or not holdout.strip():
            raise ValueError("airport split requires an explicit holdout airport ID")
        tr = indices[data.airports != holdout]
        va = np.array([], dtype=int)
        te = indices[data.airports == holdout]
    else:
        raise ValueError("unknown protocol")
    if not len(tr) or not len(te):
        raise ValueError("empty train or test split")
    return tr, va, te
