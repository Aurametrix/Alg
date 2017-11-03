all_rows = worksheet.get_all_records()

performances = []
processed_rows = []

for idx, row in enumerate(all_rows):
    if row['Processed'] != '*':
        keys = ('Artist', 'Date', 'Venue', 'Source',
                'Notes', 'Important', 'Ad', 'Contributor')

        # Create new Dict with only keys from list (Drop 'Processed')
        new_row = {key: row[key] for key in keys}
        performances.append(new_row)

        # add 2 to index to skip header row *and* convert from
        # zero-based indexing to 1-based
        processed_rows.append(idx + 2)
        
# Get a timestamp to mark output files uniquely
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

# Output JSON for ingested data
output_file = open('{}-performances.json'.format(st), 'w')
output_file.write(json.dumps(performances))

# Output row numbers for processed data in case automated
# spreadsheet update fails
output_file = open('{}-processed-rows.json'.format(st), 'w')
output_file.write(json.dumps(processed_rows))

for row in processed_rows:
    worksheet.update_cell(row, 9, '*')
