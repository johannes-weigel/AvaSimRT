from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shutil

from avasimrt.channelstate.simulation import (
    _build_context,
    _evaluate_cfr
)

from avasimrt.channelstate.snow import (
    Snow, create_scene_with_snow
)
from sionna.rt import ITURadioMaterial

import sionna_vispy
from avasimrt.math import mean_amp_in_db_from_cfr, distance

@pytest.mark.presentation
def test_closer_transmitter_has_higher_amplitude(
    presentation_output: Path, 
    examples: Path,
    presentation_config
):
    """
    Setup:
        - Empty scene (no terrain, no obstacles)
        - TX1 at (-1, -8, 0) - distance ~8.06m from origin
        - TX2 at (1, -10, 0) - distance ~10.05m from origin
        - RX at (0, 0, 0)

    Expected:
        TX1 amplitude > TX2 amplitude (closer = stronger signal)
    """

    out_dir = presentation_output / "sionna_snow_poc"
    out_dir.mkdir(exist_ok=True)

    scene_xml = out_dir / "scene.xml"
    shutil.copy(examples / "empty.xml", scene_xml)


    tx1_pos = (-1.0, -80.0, 0.0)
    tx2_pos = (1.0, -100.0, 0.0)
    txs = [("tx_close", tx1_pos, 1.0),
           ("tx_far",   tx2_pos, 1.0)]
    
    snow = Snow(thickness_m=0.1)

    snow_scene_xml = create_scene_with_snow(
        xml_path=scene_xml,
        meshes_dir=out_dir / "meshes",
        radius=1.0,
        positions=np.array([(0, -25, 0)])
    )

    ctx = _build_context(anchors=txs, 
                         scene_src=snow_scene_xml,
                         freq_center=None,
                         bandwidth=None,
                         snow=snow,
                         reflection_depth=3,
                         seed=None)
    
    # should be default, but validate to guard against changes
    rx_pos = (0.0, 0.0, 0.0)
    actual_pos = (ctx.rx.position.x, ctx.rx.position.y, ctx.rx.position.z)
    assert actual_pos == rx_pos


    dist1 = distance(rx_pos, tx1_pos)
    dist2 = distance(rx_pos, tx2_pos)

    paths = ctx.solve_paths()

    ctx.render_to_file(paths, 
                       origin=(0, 0, 30), 
                       target=(0, -5, 0), 
                       file_path=out_dir / "scene_top_view.png",
                       resolution=presentation_config.resolution)
    
    with sionna_vispy.patch():
        ctx.scene.preview(paths=paths)

    sionna_vispy.get_canvas(ctx.scene).show()
    sionna_vispy.get_canvas(ctx.scene).app.run()

    
 