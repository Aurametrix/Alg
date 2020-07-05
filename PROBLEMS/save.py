a_file = open("test.txt", "w")
for row in an_array:
    np.savetxt(a_file, row)

a_file.close()
