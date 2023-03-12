#!/bin/env python3

from random import randrange, choice
from string import Template
import markdown2
import pdfkit
import os, sys
import logging
import argparse
import PyPDF2
import aspose.words as aw

def gen_equations(num=1000, start=1, end=100, operators=['+', '-'], order=False):
    num_of_digits = 2
    i = 1
    line= ''
    while i < num:
        x = randrange(start, end)
        y = randrange(start, end)
        operator = choice(operators)
        if operator == 'x' and order and x > y:
            continue

        equation = f'{x:>{num_of_digits}d} {operator} {y:>{num_of_digits}d}'
        if operator == 'x':
            equation = equation.replace(' ', '&nbsp;')
            line = f'## ({i:>2d}) {equation} = \n'
            i = i + 1
            yield i, line
            continue

        if eval(equation) >= 0:
            equation = equation.replace(' ', '&nbsp;')
            line = f'## ({i:>2d}) {equation} = \n'
            i = i + 1
            yield i, line

    #return i, line

column_temp = Template('''
<!DOCTYPE html>
<html>
<body>
<style>
.container {
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}
h1 {
    color: #333;
    font-family: "Courier New", monospace;
    font-size: 45px;
    text-align: left;
}
h2 {
    color: #333;
    font-family: "Courier New", monospace;
    font-size: 35px;
    text-align: left;
}
ch1 {
    font-size: 50px;
    color: #333;
    font-family: Arial, sans-serif;
    text-align: center;
    margin-top: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 10vh;
    background-color: lightblue;
}

* {
    box-sizing: border-box;
}

/* Create two unequal columns that floats next to each other */
.column {
    float: left;
    padding: 10px;
    height: 300px; /* Should be removed. Only for demonstration */
}

.left {
    width: 50%;
}

.right {
    width: 50%;
}

/* Clear floats after the columns */
.row:after {
    content: "";
    display: table;
    clear: both;
}
</style>



<h1 style="color: #5e9ca0; text-align: center;">Math Day <span style="color: #00ccff;">$title</span></h1>
<h2 style="color: #2e6c80; text-align: left;">Name:&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color: #00ff00;">Score:</span></h2>

<div class="row">
  <div class="column left">
    <p>$column1</p>
  </div>
  <div class="column right">
    <p>$column2</p>
  </div>
</div>

</body>
</html>
''')

def multiple_html_to_pdf(path):
    """ converts multiple html files to a single pdf
    args: path to directory containing html files
    """
    empty_html = '<html><head></head><body></body></html>'
    for file in os.listdir(path):
        if file.endswith(".html"):
            print(file)
            # append html files
            with open(path + file, 'r') as f:
                html = f.read()
                empty_html = empty_html.replace('</body></html>', html + '</body></html>')
    return empty_html
    ## save merged html
    #with open('merged.html', 'w') as f:
    #    f.write(empty_html)

def main(opts):
    html_pages = []
    for day in range(1, opts.numofdays + 1, 1):
        with open(f'mathDay{day}.html', 'w+') as f:
            title = f'{day}'
            content1 = ''
            content2 = ''
            rows = 16
            i = 0
            for _, line in gen_equations(start=opts.start, end=opts.end):
                if i < rows:
                    content1 += line
                    content1 = markdown2.markdown(content1, extras=["break-on-newline",
                        "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables",
                        "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])
                else:
                    content2 += line
                    content2 = markdown2.markdown(content2, extras=["break-on-newline",
                        "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables",
                        "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])
                i = i + 1
                if i >= rows * 2:
                    break
            #print(content1)
    
            html_str = column_temp.safe_substitute(column1=content1, column2=content2, title=title)
            html_pages.append(html_str)

        if not opts.multiply:
            continue

        with open(f'mathDay{day}m.html', 'w+') as f:
            title = f'{day}'
            content1 = ''
            content2 = ''
            rows = 16
            i = 0
            for _, line in gen_equations(start=1, end=10, operators=['x'], order=True):
                if i < rows:
                    content1 += line
                    content1 = markdown2.markdown(content1, extras=["break-on-newline",
                        "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables",
                        "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])
                else:
                    content2 += line
                    content2 = markdown2.markdown(content2, extras=["break-on-newline",
                        "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables",
                        "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])
                i = i + 1
                if i >= rows * 2:
                    break
            #print(content1)
    
            html_str = column_temp.safe_substitute(column1=content1, column2=content2, title=title)
            html_pages.append(html_str)

            f.write(html_str)

    gen_pdf(html_pages)
    return html_pages

def gen_pdf(html_pages):
    #    output = aw.Document()
    #    # Remove all content from the destination document before appending.
    #    output.remove_all_children()
    #    for filename in html_pages:
    #        input = aw.Document(filename)
    #        # Append the source document to the end of the destination document.
    #        output.append_document(input, aw.ImportFormatMode.KEEP_SOURCE_FORMATTING)
    #
    #    output.save("Output.pdf")

    with open('MathTest.pdf', 'wb') as f:
        config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
        pdfWriter = PyPDF2.PdfWriter()
        for page in html_pages:
            pdfkit.from_string(page, '/tmp/tmp.pdf', configuration=config)
            input_pdf = PyPDF2.PdfReader('/tmp/tmp.pdf')
            pdf_page = input_pdf.pages[0]
            pdfWriter.add_page(pdf_page)

        pdfWriter.write(f)


def create_arg_parser(prog):
    parser = argparse.ArgumentParser(prog=prog,
            description="Offline Diags Formatted Test Script")
    parser.add_argument('-d', '--debug', help=f"Enable {prog} debug mode", 
            action='store_true', default=False)
    parser.add_argument('-s', '--start', help="start number",
            metavar='START', type=int, default=1)
    parser.add_argument('-e', '--end', help="end number", 
            metavar='END', type=int, default=100)
    parser.add_argument('-m', '--multiply', help="with multiply", 
            action='store_true', default=False)
    parser.add_argument('-n', '--numofdays', help="Number of Days", 
            metavar='DAYS', type=int, default=10)
    parser.set_defaults(myfunc=main)

    return parser

if __name__ == '__main__':
    prog = 'genMath'
    parser = create_arg_parser(prog)
    params = parser.parse_args()
    print(params)

    fn = getattr(params, "myfunc", None)

    if fn is None:
        parser.print_usage()
    else:
        fn(params)
    
