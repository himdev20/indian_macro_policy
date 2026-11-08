"""`python -m indiapolicy` reproduces every figure, table and number in the report."""
from .pipeline import run

if __name__ == "__main__":
    print("Reproducing ECO612 analysis ...")
    run()
    print("Done -> output/figures, output/tables, output/results.json")
