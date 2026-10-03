"""最小探查:body-stats schema + head。"""

import pyarrow as pa
import pyarrow.feather as feather

F = r"D:\dsh\autoaim\.cache\body-stats-male-cns-v1.0-minconf-0.5.feather"
with pa.memory_map(F, "r") as src:
    r = pa.ipc.open_file(src)
    print("num_record_batches", r.num_record_batches)
    print(r.schema)
    nrows = 0
    for i in range(r.num_record_batches):
        nrows += r.get_batch(i).num_rows
    print("total rows", nrows)
    b = r.get_batch(0)
    print(b.slice(0, 5).to_pandas().to_string())
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    print(b.slice(0, 5).to_pandas().to_string())
