from pymatgen.ext.matproj import MPRester, TaskType
import os
material_ids = ["mp-32800", "mp-23494"]
task_types = [TaskType.GGA_OPT, TaskType.GGA_UNIFORM]
file_patterns = ["vasprun*", "OUTCAR*"]
with MPRester(os.environ["MP_API_KEY"]) as mpr:
    meta, urls = mpr.get_download_info(
        material_ids, task_types=task_types, file_patterns=file_patterns
    )

print(meta)
print(urls)
