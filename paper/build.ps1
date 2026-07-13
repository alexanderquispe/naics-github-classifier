# build.ps1 - one-command rebuild of the NAICS-GH paper.
#
# Steps:
#   1. Regenerate validation tables from the CSVs.
#   2. Convert DATASHEET.md to LaTeX (strips the top-level H1).
#   3. Run pdflatex / bibtex / pdflatex x2.
#
# Usage:
#   cd paper; .\build.ps1

$ErrorActionPreference = 'Stop'
$paper = $PSScriptRoot
Set-Location $paper

$python = 'C:\Users\Alexander\anaconda3\envs\naics\python.exe'

Write-Host "[1/3] Regenerating validation tables..." -ForegroundColor Cyan
& $python (Join-Path $paper 'validation\generate_tex_tables.py')
if ($LASTEXITCODE -ne 0) { throw "Table generation failed." }
& $python (Join-Path $paper 'appendix\generate_naics_table.py')
if ($LASTEXITCODE -ne 0) { throw "NAICS appendix-table generation failed." }
& $python (Join-Path $paper 'appendix\generate_prompt_appendix.py')
if ($LASTEXITCODE -ne 0) { throw "Prompt appendix generation failed." }
& $python (Join-Path $paper 'appendix\generate_schema_appendix.py')
if ($LASTEXITCODE -ne 0) { throw "Schema appendix generation failed." }
& $python (Join-Path $paper 'appendix\generate_stratification_appendix.py')
if ($LASTEXITCODE -ne 0) { throw "Stratification appendix generation failed." }
& $python (Join-Path $paper 'appendix\generate_training_appendix.py')
if ($LASTEXITCODE -ne 0) { throw "Training appendix generation failed." }

Write-Host "[2/3] Converting DATASHEET.md to LaTeX..." -ForegroundColor Cyan
$md = Get-Content (Join-Path $paper 'DATASHEET.md')
$start = 0
for ($i = 0; $i -lt $md.Count; $i++) {
    if ($md[$i] -match '^# ') { $start = $i + 1; break }
}
$body = $md[$start..($md.Count - 1)]
$tempMd = Join-Path $env:TEMP 'naics_datasheet.md'
$body | Set-Content -Encoding utf8 $tempMd
pandoc $tempMd -f markdown -t latex --wrap=preserve -o (Join-Path $paper 'datasheet_body.tex')
if ($LASTEXITCODE -ne 0) { throw "Pandoc conversion failed." }

Write-Host "[3/3] Compiling LaTeX (4 passes)..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed; see main.log." }
bibtex main | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pdflatex final pass failed; see main.log." }

$pdf = Get-Item (Join-Path $paper 'main.pdf')
$kb = [math]::Round($pdf.Length / 1KB, 1)
Write-Host ""
Write-Host "[OK] $($pdf.Name)  $kb KB  $($pdf.LastWriteTime)" -ForegroundColor Green
