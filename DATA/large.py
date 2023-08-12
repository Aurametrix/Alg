# large file is split into smaller files
# small file with a string is extracted from large file 

def split_text_file(file_path, lines_per_chunk, search_string):
    total_lines = 0
    chunk_number = 0
    lines = []
    lines_with_string = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_lines += 1
            if search_string in line:
                lines_with_string.append(line)
            lines.append(line)
            if len(lines) >= lines_per_chunk:
                with open(f'{file_path}_chunk_{chunk_number}.txt', 'w', encoding='utf-8') as chunk_file:
                    chunk_file.writelines(lines)
                lines = []
                chunk_number += 1

        # Write any remaining lines to a final chunk
        if lines:
            with open(f'{file_path}_chunk_{chunk_number}.txt', 'w', encoding='utf-8') as chunk_file:
                chunk_file.writelines(lines)

    # Write lines containing the search string to a separate file
    with open('chunk_string.txt', 'w', encoding='utf-8') as string_file:
        string_file.writelines(lines_with_string)

    print(f'{chunk_number + 1} chunks created.')
    print(f'Total lines in the file: {total_lines}')
    print(f'Lines containing "{search_string}" written to chunk_string.txt.')
    
    
file_path = 'large_file.txt'

lines_per_chunk = 100000
search_string = "search_term"

split_text_file(file_path, lines_per_chunk, search_string)

# chunk_string in the same dir as the code; other chunks where original file is
