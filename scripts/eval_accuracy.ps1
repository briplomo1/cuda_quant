# Accuracy with INT8 activations for every thresholds file from scripts/summarize.py.
# Run locally from the project folder (needs data/ with the model and CIFAR-10):
#   powershell -File scripts/eval_accuracy.ps1
# Writes results/raw/accuracy.txt: one "name accuracy" line per thresholds file.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$out = "results/raw/accuracy.txt"
Set-Content $out "" -Encoding ascii
foreach ($f in Get-ChildItem results/thresholds/*.txt | Sort-Object Name) {
    $line = py -3.13 prepare.py --eval $f.FullName | Select-String "accuracy with INT8"
    $acc = [regex]::Match($line.ToString(), "([\d.]+)%").Groups[1].Value
    "$($f.BaseName) $acc" | Add-Content $out -Encoding ascii
    "$($f.BaseName): $acc%"
}
