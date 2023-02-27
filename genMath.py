#!/bin/env python3

from random import randrange, choice
from string import Template
import markdown2
import pdfkit
import sys

def gen_equations(num=25, numrange=100, operators=['+', '-'], index_start=1, with_answer=False, order=False):
    num_of_digits = 2
    i = index_start
    line= ''
    while True:
        x = randrange(1, numrange)
        y = randrange(1, numrange)
        operator = choice(operators)
        equation = f'{x:>{num_of_digits}d} {operator} {y:>{num_of_digits}d}'
        if eval(equation) >= 0:
            if with_answer: 
                answer = eval(equation)
            else:
                answer = ''
            equation = equation.replace(' ', '&nbsp;')
            line += f'## ({i:>2d}) {equation} = {answer}\n'
            if (i >= num):
                break
            i = i + 1
    #print(line)
    return i, line

column_temp = Template('''
<!DOCTYPE html>
<html>
<head>
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
    font-size: 28px;
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

</head>


<body>
<p><ch1>$title</ch1></p>

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

if len(sys.argv) == 2:
    num_max = int(sys.argv[1])
else:
    num_max = 100

for day in range(1,10,1):
    with open(f'mathDay{day}.html', 'w+') as f:
        title = f'Day{day}'
        i_end, content1 = gen_equations(num=13, numrange=num_max, index_start=1)
        content1 = markdown2.markdown(content1, extras=["break-on-newline", "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables", "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])
        i_end, content2 = gen_equations(num=25, numrange=num_max, index_start=i_end + 1)
        content2 = markdown2.markdown(content2, extras=["break-on-newline", "code-friendly", "cuddled-lists", "fenced-code-blocks", "tables", "footnotes", "smarty-pants", "numbering	", "tables", "strike", "spoiler"])

        html_str = column_temp.safe_substitute(column1=content1, column2=content2, title=title)

        f.write(html_str)



