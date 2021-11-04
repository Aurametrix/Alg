[Code Examples](https://justshowmethecode.com/browse-code-examples/python%20regex)


# Python Enhancement Protocol (PEP) 

[Acceptance of Pattern Matching PEPs 634, 635, 636, Rejection of PEPs 640 and 642](https://mail.python.org/archives/list/python-dev@python.org/thread/SQC2FTLFV5A7DV7RCEAR2I2IKJKGK7W3/)

[Pattern matching accepted for Python](https://lwn.net/Articles/845480/)

https://www.python.org/dev/peps/pep-0634/
https://www.python.org/dev/peps/pep-0635/
https://www.python.org/dev/peps/pep-0636/



"Given a pattern and a string input - find if the string follows the same pattern and return true or false."

Examples:

Pattern : "abba", input: "redbluebluered" should return 1.
Pattern: "aaaa", input: "asdasdasdasd" should return 1.
Pattern: "aabb", input: "xyzabcxzyabc" should return 0.


Regular expressions (abbreviated as regex or regexp, with plural forms regexes, regexps, or regexen) are written in a formal language that can be interpreted by a regular expression processor, a program that either serves as a parser generator or examines text and identifies parts that match the provided specification.

Accepted:

Specification https://www.python.org/dev/peps/pep-0634/

Motivation and Rationale https://www.python.org/dev/peps/pep-0635/

Tutorial https://www.python.org/dev/peps/pep-0636/

Rejected:

Unused variable syntax https://www.python.org/dev/peps/pep-0640/

Explicit Pattern Syntax for Structural Pattern Matching https://www.python.org/dev/peps/pep-0642/
