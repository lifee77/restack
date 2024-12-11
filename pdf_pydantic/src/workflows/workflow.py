from restack_ai.workflow import workflow, import_functions, log
from datetime import timedelta

with import_functions():
    from pydantic import BaseModel
    from src.functions.ocr import ocr, FunctionInput

class WorkflowInput(BaseModel):
    file_type: str
    file_binary:str

@workflow.defn()
class PdfWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInput):
        log.info("PdfWorkflow started")

        ocr_result = await workflow.step(
            ocr,
            FunctionInput(
                file_type=input.file_type,
                file_binary=input.file_binary
            ),
            start_to_close_timeout=timedelta(seconds=120)
        )
        log.info("PdfWorkflow completed")
        return ocr_result
