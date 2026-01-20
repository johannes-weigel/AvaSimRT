from __future__ import annotations

import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path


logger = logging.getLogger(__name__)

def _get_blender_export_script(
    obj_output: Path | None,
    xml_output: Path | None,
    ply_output: Path | None,
) -> str:
    return textwrap.dedent(f'''\
        import bpy
        import sys

        # Get mesh objects only (excluding cameras, lights, empties, etc.)
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

        if len(mesh_objects) == 0:
            print("ERROR: No mesh objects found in the scene", file=sys.stderr)
            sys.exit(1)

        if len(mesh_objects) > 1:
            print(f"ERROR: Expected exactly 1 mesh object, found {{len(mesh_objects)}}: {{[o.name for o in mesh_objects]}}", file=sys.stderr)
            sys.exit(1)

        obj = mesh_objects[0]
        print(f"Found single mesh object: {{obj.name}}")

        # Deselect all, then select only our object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Export OBJ
        obj_output = {repr(str(obj_output)) if obj_output else 'None'}
        if obj_output:
            print(f"Exporting OBJ to: {{obj_output}}")
            bpy.ops.wm.obj_export(
                filepath=obj_output,
                export_selected_objects=True,
                export_materials=True,
                export_triangulated_mesh=True,
                forward_axis='Y',
                up_axis='Z',
            )
            print("OBJ export complete")

        # Export PLY (for Mitsuba XML to reference)
        ply_output = {repr(str(ply_output)) if ply_output else 'None'}
        if ply_output:
            print(f"Exporting PLY to: {{ply_output}}")
            bpy.ops.wm.ply_export(
                filepath=ply_output,
                export_selected_objects=True,
                forward_axis='Y',
                up_axis='Z',
            )
            print("PLY export complete")

        # Export Mitsuba XML
        xml_output = {repr(str(xml_output)) if xml_output else 'None'}
        if xml_output:
            print(f"Exporting Mitsuba XML to: {{xml_output}}")
            try:
                bpy.ops.export_scene.mitsuba(filepath=xml_output)
                print("Mitsuba XML export complete")
            except AttributeError:
                print("ERROR: Mitsuba addon not available. Install mitsuba-blender addon.", file=sys.stderr)
                sys.exit(2)

        print("All exports complete")
    ''')


def run_blender_export(
    blend_path: Path,
    obj_output: Path,
    xml_output: Path,
    ply_output: Path,
) -> None:
    script = _get_blender_export_script(obj_output, xml_output, ply_output)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = Path(f.name)

    try:
        cmd = [
            'blender',
            '--background',
            str(blend_path),
            '--python', str(script_path),
        ]
        logger.info(f"Running Blender export: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            if result.returncode == 2:
                raise RuntimeError(
                    f"Mitsuba addon not installed in Blender. "
                    f"Install it from: https://github.com/mitsuba-renderer/mitsuba-blender\n"
                    f"Blender output: {error_msg}"
                )
            raise RuntimeError(
                f"Blender export failed with code {result.returncode}:\n{error_msg}"
            )

        logger.debug(f"Blender output:\n{result.stdout}")

    finally:
        script_path.unlink(missing_ok=True)
