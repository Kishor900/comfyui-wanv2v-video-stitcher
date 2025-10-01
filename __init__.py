# Wanv2vControlnetManager package entrypoint (with error logging)
import os, traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .numbered_batch_debugger import NumberedBatchDebugger
    from .wanv2v_iter_control_group import WanV2VIterControlGroup
    from .wanv2v_merge_deoverlap import WanV2VMergeDeOverlap

    NODE_CLASS_MAPPINGS.update({
        "NumberedBatchDebugger": NumberedBatchDebugger,
        "WanV2VIterControlGroup": WanV2VIterControlGroup,
        "WanV2VMergeDeOverlap": WanV2VMergeDeOverlap,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "NumberedBatchDebugger": "WanV2V Debug: Number Batch",
        "WanV2VIterControlGroup": "WanV2V: Iter Control Group",
        "WanV2VMergeDeOverlap": "WanV2V: Merge De-Overlap (10x)",
    })

    __all__ = ["NumberedBatchDebugger", "WanV2VIterControlGroup", "WanV2VMergeDeOverlap"]

except Exception:
    # Write a readable error file beside this package
    err_path = os.path.join(os.path.dirname(__file__), "Wanv2v_load_error.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    # Re-raise so ComfyUI prints it in the console too
    raise
