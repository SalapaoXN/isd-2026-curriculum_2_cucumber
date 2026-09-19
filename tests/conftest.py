"""Legacy import shims: original tests import top-level modules.

The clean tree moved code under src.pipeline.*; these aliases keep every
preserved test runnable unchanged (behavior verification).
"""
import sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ALIASES = {
    "prepare_data": "src.pipeline.tools.preparation.tool",
    "extract": "src.pipeline.tools.extraction.tool",
    "merge_consecutive": "src.pipeline.tools.merge.consolidator",
    "llm_spell_corrector": "src.pipeline.tools.correction.corrector",
    "evaluate": "src.pipeline.tools.evaluation.evaluate",
    "evaluate_curriculum_layers": "src.pipeline.tools.evaluation.layers",
    "config": "src.pipeline.config",
    "ocr": "src.pipeline.tools.ocr.pipeline_runner",
    "extract_rules": "src.pipeline.tools.extraction.rule_cli",
    "map_rules_policy": "src.pipeline.tools.merge.policy_cli",
    "pipeline": "src.pipeline.run",
    "ask": "scripts.ask",
}

def _alias(old, new):
    try:
        mod = __import__(new, fromlist=["*"])
        sys.modules.setdefault(old, mod)
    except Exception:
        pass

for _o, _n in _ALIASES.items():
    _alias(_o, _n)

# src.* legacy paths used by some tests
try:
    import src.pipeline.utils.file_handler as _fh
    sys.modules.setdefault("src.file_handler", _fh)
    import src.pipeline.utils.page_metadata as _pm
    sys.modules.setdefault("src.page_metadata", _pm)
    import src.pipeline.utils.pre_clean as _pc
    sys.modules.setdefault("src.pre_clean", _pc)
    import src.pipeline.tools.ocr.engine as _oe
    sys.modules.setdefault("src.ocr_engine", _oe)
    import src.pipeline.tools.ocr.pipeline_runner as _rp
    sys.modules.setdefault("src.run_pipeline", _rp)
    import src.pipeline.config as _cfg
    sys.modules.setdefault("src.pipeline_config", _cfg)
    import src.pipeline.tools.extraction.engine as _ee
    sys.modules.setdefault("src.extractor", _ee)
    import src.pipeline.tools.extraction.rules as _re
    sys.modules.setdefault("src.rule_extractor", _re)
    import src.pipeline.tools.merge.policy as _mp
    sys.modules.setdefault("src.rules_policy_mapper", _mp)
except Exception:
    pass
