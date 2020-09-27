import PyPDF2
mypdf = open('D:\Lorem-Ipsum.pdf', mode='rb')
pdf_document = PyPDF2.PdfFileReader(mypdf)
pdf_document.numPages
first_page = pdf_document.getPage(0)
print(first_page.extractText())

#### or use tabula
import tabula

# Read pdf into list of DataFrame
df = tabula.read_pdf("test.pdf", pages='all')

# Read remote pdf into list of DataFrame
df2 = tabula.read_pdf("https://github.com/tabulapdf/tabula-java/raw/master/src/test/resources/technology/tabula/arabic.pdf")

# convert PDF into CSV file
tabula.convert_into("test.pdf", "output.csv", output_format="csv", pages='all')

# convert all PDFs in a directory
tabula.convert_into_by_batch("input_directory", output_format='csv', pages='all')
