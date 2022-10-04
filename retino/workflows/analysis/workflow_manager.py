"""Analysis Workflow Manager."""

from ..base import WorkflowManager


class AnalysisWorkflowManager(WorkflowManager):
    """Manager for Analysis Workflow."""

    workflow_name = "analysis"
    input_fields = ["sub_id", "sequence", "preproc_code", "denoise_code"]

    def _build_files(self, wf):
        ...


class RetinoAnalysisWorkflowManager(AnalysisWorkflowManager):
    ...


class FirstLevelStats(AnalysisWorkflowManager):
    ...