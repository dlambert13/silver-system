for /l %%x in (0, 1, 2) do (
    python process.py 11 %%x
    python discrepancy.py 11 %%x
)
