#!/usr/bin/env bash
# Build the distributable documents from the markdown sources.
#
#   ./docs/build_docs.sh [outdir]        default outdir: ../  (beside the repo)
#
# Produces:
#   Visitor-Counting-Console-Features-and-Specs.docx / .pdf
#   Visitor-Counting-Console-Deck.pptx / .pdf
#   Visitor-Counting-Console-Demo-Runbook.pdf
#
# Needs pandoc, and tectonic for the PDFs (brew install pandoc tectonic).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-..}"
mkdir -p "$OUT"

# A handful of glyphs the default LaTeX font cannot draw. DOCX and PPTX carry
# them fine, so this substitution is applied on the PDF path only, and to a
# temporary copy - the markdown sources keep the symbols.
pdf_safe() {
    sed -e 's/⚙/[settings]/g' -e 's/⤢/[expand]/g' -e 's/⤡/[collapse]/g' \
        -e 's/●/*/g'          -e 's/▸/>/g'        -e 's/✓/yes/g' \
        -e 's/♂/M/g'          -e 's/♀/F/g'        -e 's/–/-/g' "$1"
}

say() { printf '  %-52s %s\n' "$1" "$2"; }

# ---- features and specs: DOCX + PDF ----
pandoc docs/features-and-specs.md -o "$OUT/Visitor-Counting-Console-Features-and-Specs.docx"
say "Features-and-Specs.docx" "ok"

pdf_safe docs/features-and-specs.md > /tmp/_specs.md
pandoc /tmp/_specs.md -o "$OUT/Visitor-Counting-Console-Features-and-Specs.pdf" \
    --pdf-engine=tectonic
say "Features-and-Specs.pdf" "ok"

# ---- deck: PPTX + PDF ----
pandoc docs/slides.md -o "$OUT/Visitor-Counting-Console-Deck.pptx" --slide-level=1
say "Deck.pptx" "ok"

pdf_safe docs/slides.md > /tmp/_slides.md
pandoc /tmp/_slides.md -o "$OUT/Visitor-Counting-Console-Deck.pdf" \
    -t beamer --slide-level=1 --pdf-engine=tectonic \
    -V theme:default -V colortheme:seahorse -V fontsize:11pt
say "Deck.pdf" "ok"

# ---- demo runbook: PDF ----
pdf_safe DEMO_RUNBOOK.md > /tmp/_runbook.md
pandoc /tmp/_runbook.md -o "$OUT/Visitor-Counting-Console-Demo-Runbook.pdf" \
    --pdf-engine=tectonic --toc --toc-depth=2 \
    -V papersize:a4 -V geometry:margin=2.2cm -V fontsize:11pt \
    -V colorlinks:true -V title:"Demo runbook" -V subtitle:"Visitor Counting Console"
say "Demo-Runbook.pdf" "ok"

rm -f /tmp/_specs.md /tmp/_slides.md /tmp/_runbook.md
echo "  -> $(cd "$OUT" && pwd)"
