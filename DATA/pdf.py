import PyPDF2
mypdf = open('D:\Lorem-Ipsum.pdf', mode='rb')
pdf_document = PyPDF2.PdfFileReader(mypdf)
pdf_document.numPages
first_page = pdf_document.getPage(0)
print(first_page.extractText())
