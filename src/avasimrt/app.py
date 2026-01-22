from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

from .config import SimConfig
from .result import SimResult

from .preprocessing.preprocessor import prepare
from .motion.simulation import simulate_motion
from .motion.cache import load_all_trajectories, save_all_trajectories
from .motion.visualization import save_all_trajectory_visualizations
from .channelstate.simulation import estimate_channelstate
from .channelstate.cache import load_all_channelstates, save_all_channelstates
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


def run(config: SimConfig, blender_cmd: str | None = None) -> SimResult:
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
            scene_meshes=config.scene_meshes,
            blender_cmd=blender_cmd,
            nodes=config.nodes,
            anchors=config.anchors,
            heightmap_npy=config.heightmap_npy,
            heightmap_resolution=config.heightmap_resolution,
        )
    
        out_dir = preprocessing_result.out_dir
        run_id = preprocessing_result.run_id
        nodes = preprocessing_result.nodes
        anchors = preprocessing_result.anchors
        scene_obj = preprocessing_result.scene_obj
        scene_xml = preprocessing_result.scene_xml
        heightmap = preprocessing_result.heightmap


        # 1) MOTION (PyBullet)
        if config.trajectory_cache_dir is not None:
            with log_step("MOTION (from cache)"):
                trajectories = load_all_trajectories(config.trajectory_cache_dir)
        else:
            if len(nodes) == 0:
                return SimResult(
                    successful=True,
                    message="Gracefully aborted after preprocessing: no node configured",
                    run_id=run_id,
                    output_dir=out_dir,
                )
            with log_step("MOTION"):
                trajectories = simulate_motion(
                    cfg=config.motion,
                    nodes=nodes,
                    scene_obj=scene_obj
                )
                if config.trajectory_save:
                    trajectory_dir = out_dir / "trajectories"
                    save_all_trajectories(trajectories, trajectory_dir)

        if len(trajectories.values()) == 0:
            return SimResult(
                successful=True,
                message="Gracefully aborted after motion: no trajectories configured",
                run_id=run_id,
                output_dir=out_dir,
            )

        if config.trajectory_plots_png or config.trajectory_plots_html:
            with log_step("TRAJECTORY VISUALIZATION"):
                save_all_trajectory_visualizations(
                    trajectories,
                    out_dir / "trajectory_plots",
                    png=config.trajectory_plots_png,
                    html=config.trajectory_plots_html,
                    terrain_obj_path=scene_obj,
                    anchors=anchors,
                )

        if len(anchors) == 0:
            return SimResult(
                successful=True,
                message="Gracefully aborted after motion: no anchors configured",
                run_id=run_id,
                output_dir=out_dir,
            )

        # 2) CHANNELSTATE (Sionna RT)
        if config.channelstate_cache_dir is not None:
            with log_step("CHANNELSTATE (from cache)"):
                all_results = load_all_channelstates(
                    config.channelstate_cache_dir, trajectories=trajectories
                )
        else:
            with log_step("CHANNELSTATE"):
                channelstate_result = estimate_channelstate(
                    cfg=config.channelstate,
                    anchors=anchors,
                    trajectories=trajectories,
                    heightmap=heightmap,
                    scene_xml=scene_xml,
                    out_dir=out_dir
                )
                all_results = channelstate_result.samples
                if config.channelstate_save:
                    channelstate_dir = out_dir / "channelstate"
                    save_all_channelstates(
                        all_results,
                        channelstate_dir,
                        cfg=config.channelstate,
                        trajectories=trajectories,
                        anchors=anchors,
                        scene_xml=scene_xml,
                        durations=channelstate_result.durations,
                        total_duration=channelstate_result.total_duration,
                    )

        # 3) REPORTING (CSV export)
        csv_paths: list[Path] = []
        if config.reporting.enabled and config.reporting.csv:
            with log_step("REPORTING"):
                for node_id, results in all_results.items():
                    csv_path = out_dir / f"results_{node_id}.csv"
                    export_simresult_to_csv(results, csv_path)
                    csv_paths.append(csv_path)

        # 4) VISUALIZATION (optional)
        if config.visualization.save_all_plots:
            with log_step("VISUALIZATION_SAVE_ALL"):
                for node_id, results in all_results.items():
                    save_all_visualizations(results, out_dir / f"visualizations_{node_id}")
        if config.visualization.interactive_plots:
            
            first_results = next(iter(all_results.values()), [])
            if first_results:
                with log_step("VISUALIZATION_INTERACTIVE"):
                    interactive_visualization_shell(first_results)

        msg = "run completed"
        if csv_paths:
            msg += f" (csv: {len(csv_paths)} files)"

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
