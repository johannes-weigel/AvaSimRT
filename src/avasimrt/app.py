from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

from .config import SimConfig
from .result import SimResult

from .preprocessing.preprocessor import prepare
from .motion.simulation import simulate_motion
from .channelstate.simulation import estimate_channelstate
from .reporting.csv import export_simresult_to_csv
from .visualization.interactive import interactive_visualization_shell
from .visualization.plots import save_all_visualizations

logger = logging.getLogger(__name__)


@contextmanager
def log_step(name: str):
    start = time.perf_counter()
    logger.info("=== %s: START ===", name)
    try:
        yield
    except Exception:
        end = time.perf_counter()
        logger.error("=== %s: FAILED after %.2f s ===", name, end - start)
        logger.exception("Error in step %s", name)
        raise
    else:
        end = time.perf_counter()
        logger.info("=== %s: DONE in %.2f s ===", name, end - start)


def run(config: SimConfig) -> SimResult:
    """Run the full simulation pipeline (motion -> channelstate -> reporting/visualization)."""
    run_id: str | None = None
    out_dir: Path | None = None
    
    try:
        preprocessing_result = prepare(
            out_base=config.output,
            run_id=config.run_id,
            delete_existing=config.delete_existing,
            scene_blender=config.scene_blender,
            scene_obj=config.scene_obj,
            scene_xml=config.scene_xml,
            node=config.node,
            anchors=config.anchors,
        )
    
        out_dir = preprocessing_result.out_dir
        run_id = preprocessing_result.run_id
        node = preprocessing_result.node
        anchors = preprocessing_result.anchors
        scene_obj = preprocessing_result.scene_obj
        scene_xml = preprocessing_result.scene_xml

        # 1) MOTION (PyBullet)
        with log_step("MOTION"):
            motion_results = simulate_motion(
                cfg=config.motion,
                node=node,
                scene_obj=scene_obj
            )

        # 2) CHANNELSTATE (Sionna RT)
        if config.channelstate is None:
            results = motion_results
        else:
            with log_step("CHANNELSTATE"):
                results = estimate_channelstate(
                    cfg=config.channelstate,
                    anchors=anchors,
                    motion_results=motion_results,
                    scene_xml=scene_xml,
                    out_dir=out_dir
                )

        # 3) REPORTING (CSV export)
        csv_path: Path | None = None
        if config.reporting.enabled and config.reporting.csv:
            with log_step("REPORTING"):
                csv_path = out_dir / "results.csv"
                export_simresult_to_csv(results, csv_path)

        # 4) VISUALIZATION (optional)
        if config.visualization.save_all_plots:
            with log_step("VISUALIZATION_SAVE_ALL"):
                save_all_visualizations(results, out_dir)
        if config.visualization.interactive_plots:
            with log_step("VISUALIZATION_INTERACTIVE"):
                interactive_visualization_shell(results)

        msg = "run completed"
        if csv_path is not None:
            msg += f" (csv: {csv_path})"

        return SimResult(
            successful=True,
            message=msg,
            run_id=run_id,
            output_dir=out_dir,
        )
    except Exception as e:
        return SimResult(
            successful=False,
            message=str(e),
            run_id=run_id,
            output_dir=out_dir,
        )
