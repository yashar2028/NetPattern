from netpattern_engine.spec.task import TaskSpec
from netpattern_engine.tasks.base import TaskPlugin
from netpattern_engine.tasks.classification import (
    MultiLabelClassification,
    Regression,
    SingleLabelClassification,
)
from netpattern_engine.tasks.detection import BoxDetection
from netpattern_engine.tasks.segmentation import SemanticSegmentation

TASKS: dict[str, type[TaskPlugin]] = {
    plugin.type: plugin
    for plugin in (
        SingleLabelClassification,
        MultiLabelClassification,
        Regression,
        SemanticSegmentation,
        BoxDetection,
    )
}


def get_task(spec: TaskSpec) -> TaskPlugin:
    return TASKS[spec.type](spec)
