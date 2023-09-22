import xlsxwriter

s = [
    20,
    40,
    50,
    70,
    80,
    100,
    110,
    130,
    150,
    160,
    180,
    200,
    220,
    230,
    250,
    25,
    45,
    60,
    75,
    90,
    105,
    120,
    135,
    155,
    170,
    185,
    205,
    225,
    240,
    255,
]

# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('hello.xlsx')

worksheet = workbook.add_worksheet()

s.sort()
customer_num = 0
for i in range(len(s)):
    tag = "Departure"
    if i % 2 == 0:
        tag = "Arrival"
        customer_num += 1

    worksheet.write(f'A{i}', s[i])
    worksheet.write(f'B{i}', customer_num)
    worksheet.write(f'C{i}', tag)



workbook.close()