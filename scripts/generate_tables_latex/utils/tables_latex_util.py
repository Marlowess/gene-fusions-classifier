#-*- encoding: utf-8 -*-

# https://miktex.org/howto/install-miktex-unx
# https://jeltef.github.io/PyLaTeX/current/index.html

import argparse
import datetime
import json
import yaml
import os
import sys
import time

from latex import build_pdf

from pylatex import Document, PageStyle, Head, Foot, MiniPage, \
    StandAloneGraphic, MultiColumn, Tabu, LongTabu, LargeText, MediumText, \
    LineBreak, NewPage, Tabularx, TextColor, simple_page_number, Package
from pylatex.utils import bold, italic, escape_latex, NoEscape

def create_example_table():
    geometry_options = {
        "head": "40pt",
        "margin": "0.5in",
        "bottom": "0.6in",
        "includeheadfoot": True
    }
    # doc = Document(geometry_options=geometry_options)
    doc = Document()
    doc.packages.append(Package('geometry', options=['tmargin=1cm',
                                                 'lmargin=10cm']))

    # Generating first page style
    # first_page = PageStyle("firstpage")
    # Add statement table
    with doc.create(LongTabu("X[l] X[2l] X[r] X[r] X[r]",
                             row_height=1.5)) as data_table:
        data_table.add_row(["date",
                            "description",
                            "debits($)",
                            "credits($)",
                            "balance($)"],
                           mapper=bold,
                           color="lightgray")
        data_table.add_empty_row()
        data_table.add_hline()
        row = ["2016-JUN-01", "Test", "$100", "$1000", "-$900"]
        for i in range(30):
            if (i % 2) == 0:
                data_table.add_row(row, color="lightgray")
            else:
                data_table.add_row(row)
    doc.generate_pdf("example_report", clean_tex=False, compiler='pdflatex')
    pass